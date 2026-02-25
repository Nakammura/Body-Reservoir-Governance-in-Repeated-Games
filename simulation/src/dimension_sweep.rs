// ==========================================================================
// dimension_sweep.rs — Reservoir dimension (richness) sweep
//
// Tests the hypothesis that richer reservoirs (higher d) produce better
// body governance: lower KL, lower variance, and higher optimal α*.
//
// Protocol:
//   For each d ∈ {5, 10, 15, 20, 30, 50, 75, 100}:
//     Phase 1: Train readout
//     Phase 2: Habituate
//     Phase 3: Sweep α, measure KL, variance, payoff, convergence time
//
// Output CSV:
//   d,alpha,seed,kl_noisy,mean_payoff,mean_action,mean_body_out,action_var,convergence_time
// ==========================================================================

mod esn;
use esn::*;

fn payoff(a_i: f64, a_j: f64) -> f64 {
    let r = 3.0;
    let s = 0.0;
    let t = 5.0;
    let p = 1.0;
    r * a_i * a_j + s * a_i * (1.0 - a_j) + t * (1.0 - a_i) * a_j + p * (1.0 - a_i) * (1.0 - a_j)
}

fn warmup_cooperative(esn: &mut ESN, n_steps: usize, sigma_xi: f64, rng: &mut Rng) {
    let inp = vec![1.0, 1.0];
    for _ in 0..n_steps {
        esn.step(&inp, sigma_xi, rng);
    }
}

fn noisy_coop_schedule(n: usize, epsilon: f64, rng: &mut Rng) -> Vec<f64> {
    (0..n).map(|_| {
        if rng.next_f64() < epsilon { 0.0 } else { 1.0 }
    }).collect()
}

/// Coupled BRG dynamics, returns (states, body_outs, actions).
fn run_coupled(
    esn: &mut ESN,
    alpha: f64,
    w_out: &Vec1D,
    b_out: f64,
    opp_actions: &[f64],
    sigma_xi: f64,
    rng: &mut Rng,
) -> (Vec<Vec1D>, Vec<f64>, Vec<f64>) {
    let n = opp_actions.len();
    let mut states = Vec::with_capacity(n);
    let mut body_outs = Vec::with_capacity(n);
    let mut actions = Vec::with_capacity(n);
    let mut last_opp = opp_actions[0];

    for t in 0..n {
        let a_opp = opp_actions[t];
        let a_body = body_readout(&esn.state, w_out, b_out);
        let a_cog = last_opp;
        let action = alpha * a_body + (1.0 - alpha) * a_cog;
        let input = vec![action, a_opp];
        esn.step(&input, sigma_xi, rng);
        states.push(esn.state.clone());
        body_outs.push(a_body);
        actions.push(action);
        last_opp = a_opp;
    }
    (states, body_outs, actions)
}

/// Measure convergence time: first t where |body_out - final_mean| < threshold
/// for all subsequent steps.
fn convergence_time(body_outs: &[f64], threshold: f64) -> usize {
    let n = body_outs.len();
    if n < 20 { return n; }
    // Final mean from last 50 steps
    let tail = &body_outs[n.saturating_sub(50)..];
    let final_mean = tail.iter().sum::<f64>() / tail.len() as f64;

    for start in 0..n {
        let converged = body_outs[start..].iter()
            .all(|&v| (v - final_mean).abs() < threshold);
        if converged {
            return start;
        }
    }
    n
}

fn main() {
    let rho = 0.9;
    let sigma_xi = 0.15;
    let n_in = 2;

    let beta = 0.01;
    let alpha_oja = 1.0;
    let rho_min = 0.05;
    let rho_max = 0.99;
    let n_fast = 500;
    let n_habit = 200;

    let n_burnin = 500;
    let n_train = 2000;
    let n_warmup = 500;
    let target_c = 0.95;
    let target_d = 0.05;
    let lambda_reg = 0.001;

    let kl_samples = 2000;
    let epsilon = 0.1;

    let dimensions = [5, 10, 15, 20, 30, 50, 75, 100];
    let alphas: Vec<f64> = (0..=10).map(|i| i as f64 * 0.1).collect();
    let n_seeds = 20;

    println!("regime,d,alpha,seed,kl_noisy,mean_payoff,mean_action,mean_body_out,action_var,convergence_time,w_out_norm2");

    let regimes = [("scaled", true), ("fixed", false)];

    for &(regime_name, scale_lambda) in &regimes {
    for &d in &dimensions {
        eprintln!("=== {} d={} ===", regime_name, d);

        // Scale regularization with dimension (or keep fixed)
        let lambda_d = if scale_lambda {
            lambda_reg * d as f64 / 30.0
        } else {
            lambda_reg
        };

        for seed_idx in 0..n_seeds {
            let seed = 900 + seed_idx as u64 * 1000 + d as u64 * 100_000;
            let mut rng = Rng::new(seed);
            let mut esn = ESN::new(d, n_in, rho, &mut rng);

            let coop_input = vec![1.0, 1.0];
            let defect_input = vec![0.0, 0.0];

            // Phase 1: Development
            let mut esn_c = esn.clone_esn();
            let mut rng_c = Rng::new(seed + 2000);
            let coop_states = drive_and_collect(
                &mut esn_c, &coop_input, n_burnin, n_train, sigma_xi, &mut rng_c,
            );
            let mut esn_d = esn.clone_esn();
            let mut rng_d = Rng::new(seed + 3000);
            let defect_states = drive_and_collect(
                &mut esn_d, &defect_input, n_burnin, n_train, sigma_xi, &mut rng_d,
            );
            let (w_out, b_out) = train_readout_ridge(
                &coop_states, &defect_states, target_c, target_d, lambda_d,
            );

            // Compute ||w_out||²
            let w_out_norm2: f64 = w_out.iter().map(|x| x * x).sum();

            // Phase 2: Habituation
            for _ in 0..n_habit {
                let mut fast_states = Vec::with_capacity(n_fast);
                for _ in 0..n_fast {
                    esn.step(&coop_input, sigma_xi, &mut rng);
                    fast_states.push(esn.state.clone());
                }
                esn.oja_update(&fast_states, beta, alpha_oja, rho_min, rho_max);
            }

            // Reference distribution (habituated, pure cooperative, α=1)
            let mut esn_ref = esn.clone_esn();
            let mut rng_ref = Rng::new(seed + 4000);
            warmup_cooperative(&mut esn_ref, n_warmup, sigma_xi, &mut rng_ref);
            let pure_opp = vec![1.0; kl_samples];
            let (ref_states, _, _) = run_coupled(
                &mut esn_ref, 1.0, &w_out, b_out,
                &pure_opp, sigma_xi, &mut rng_ref,
            );

            // Sweep α
            for &alpha in &alphas {
                let mut esn_eval = esn.clone_esn();
                let mut rng_eval = Rng::new(seed + 7000 + (alpha * 1000.0) as u64);
                warmup_cooperative(&mut esn_eval, n_warmup, sigma_xi, &mut rng_eval);

                // Same opponent noise across alphas within seed
                let mut rng_opp = Rng::new(seed + 8000);
                let noisy_opp = noisy_coop_schedule(kl_samples, epsilon, &mut rng_opp);

                let (states_noisy, body_outs, actions) = run_coupled(
                    &mut esn_eval, alpha, &w_out, b_out,
                    &noisy_opp, sigma_xi, &mut rng_eval,
                );

                // KL(noisy || habituated reference)
                let kl_noisy = kl_knn(&states_noisy, &ref_states, 5).max(0.0);

                let mean_bo: f64 = body_outs.iter().sum::<f64>() / kl_samples as f64;
                let mean_act: f64 = actions.iter().sum::<f64>() / kl_samples as f64;
                let mean_pay: f64 = actions.iter().zip(&noisy_opp)
                    .map(|(&a, &o)| payoff(a, o)).sum::<f64>() / kl_samples as f64;
                let act_var: f64 = {
                    let m = mean_act;
                    actions.iter().map(|&a| (a - m) * (a - m)).sum::<f64>() / kl_samples as f64
                };

                // Convergence time (for α=1 self-consistent convergence)
                let conv_time = if (alpha - 1.0).abs() < 0.01 {
                    // Run a separate short convergence test with deterministic opponent
                    let mut esn_conv = esn.clone_esn();
                    let mut rng_conv = Rng::new(seed + 9000);
                    warmup_cooperative(&mut esn_conv, n_warmup, sigma_xi, &mut rng_conv);
                    let det_opp = vec![1.0; 200];
                    let (_, bo_conv, _) = run_coupled(
                        &mut esn_conv, 1.0, &w_out, b_out,
                        &det_opp, sigma_xi, &mut rng_conv,
                    );
                    convergence_time(&bo_conv, 0.01)
                } else {
                    0
                };

                println!("{},{},{:.1},{},{:.6},{:.6},{:.6},{:.6},{:.8},{},{:.8}",
                    regime_name, d, alpha, seed_idx, kl_noisy, mean_pay, mean_act, mean_bo, act_var, conv_time, w_out_norm2);
            }
            eprintln!("  {} d={} seed {} done", regime_name, d, seed_idx);
        }
    }
    } // end regimes loop
    eprintln!("Done.");
}
