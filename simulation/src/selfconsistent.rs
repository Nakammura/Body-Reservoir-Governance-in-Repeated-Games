// ==========================================================================
// selfconsistent.rs — Self-consistent fixed point & KL landscape
//
// Key demonstrations:
//   1. α=1 coupled system converges to self-consistent cooperative FP
//   2. In noisy environment (opponent defects with prob ε), body governance
//      SMOOTHS noise through reservoir dynamics → lower KL than TfT
//   3. KL(α) landscape is nonlinear, favoring high α in noisy cooperation
//
// Protocol:
//   Phase 1 — Development: Train readout on fresh reservoir
//   Phase 2 — Habituation: Oja only (preserve input sensitivity)
//   Phase 3 — Evaluation:
//     Part A: Time series showing body output convergence
//     Part B: KL landscape with noisy cooperative opponent
//
// Outputs two CSV sections (separated by ---KL_LANDSCAPE--- marker):
//   Part 1: experiment,alpha,seed,t,body_output,action,state_norm
//   Part 2: alpha,seed,kl_noisy,kl_pure,mean_body_out,mean_action,mean_payoff
// ==========================================================================

mod esn;
use esn::*;

fn payoff(a_i: f64, a_j: f64, r: f64, s: f64, t: f64, p: f64) -> f64 {
    r * a_i * a_j + s * a_i * (1.0 - a_j) + t * (1.0 - a_i) * a_j + p * (1.0 - a_i) * (1.0 - a_j)
}

/// Coupled BRG with deterministic opponent schedule.
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
        let a_cog = last_opp; // TfT
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

/// Generate noisy cooperative opponent schedule.
/// Cooperates (1.0) with prob 1-epsilon, defects (0.0) with prob epsilon.
fn noisy_coop_schedule(n: usize, epsilon: f64, rng: &mut Rng) -> Vec<f64> {
    (0..n).map(|_| {
        if rng.next_f64() < epsilon { 0.0 } else { 1.0 }
    }).collect()
}

fn warmup_cooperative(esn: &mut ESN, n_steps: usize, sigma_xi: f64, rng: &mut Rng) {
    let inp = vec![1.0, 1.0];
    for _ in 0..n_steps {
        esn.step(&inp, sigma_xi, rng);
    }
}

fn main() {
    let d = 30;
    let rho = 0.9;
    let sigma_xi = 0.15;
    let n_in = 2;

    let beta = 0.01;
    let alpha_oja = 1.0;
    let rho_min = 0.05;
    let rho_max = 0.99;
    let n_fast = 500;
    let n_habit = 200;

    let n_train = 2000;
    let n_burnin = 500;
    let n_warmup = 500;
    let target_c = 0.95;
    let target_d = 0.05;
    let lambda_reg = 0.001;

    let kl_samples = 2000;
    let epsilon = 0.1; // opponent noise level

    let n_seeds = 20;
    let alphas_fine: Vec<f64> = (0..=20).map(|i| i as f64 * 0.05).collect();

    let coop_input = vec![1.0, 1.0];
    let defect_input = vec![0.0, 0.0];

    let r = 3.0; let s_pay = 0.0; let t_pay = 5.0; let p_pay = 1.0;

    // === Part 1: Time series ===
    eprintln!("=== Part 1: Self-consistent fixed point convergence ===");
    println!("experiment,alpha,seed,t,body_output,action,state_norm");

    let ts_alphas = [0.0, 0.5, 1.0];
    let n_ts = 500;

    for seed_idx in 0..n_seeds.min(10) {
        let seed = 100 + seed_idx as u64 * 1000;
        let mut rng = Rng::new(seed);
        let mut esn = ESN::new(d, n_in, rho, &mut rng);

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
            &coop_states, &defect_states, target_c, target_d, lambda_reg,
        );

        let mean_c: f64 = coop_states.iter()
            .map(|s| body_readout(s, &w_out, b_out)).sum::<f64>() / n_train as f64;
        let mean_d: f64 = defect_states.iter()
            .map(|s| body_readout(s, &w_out, b_out)).sum::<f64>() / n_train as f64;

        // Phase 2: Habituation
        for _ in 0..n_habit {
            let mut fast_states = Vec::with_capacity(n_fast);
            for _ in 0..n_fast {
                esn.step(&coop_input, sigma_xi, &mut rng);
                fast_states.push(esn.state.clone());
            }
            esn.oja_update(&fast_states, beta, alpha_oja, rho_min, rho_max);
        }

        // Post-habituation readout check
        let mut esn_c2 = esn.clone_esn();
        let mut rng_c2 = Rng::new(seed + 4000);
        let post_coop = drive_and_collect(
            &mut esn_c2, &coop_input, n_burnin, 500, sigma_xi, &mut rng_c2,
        );
        let mean_c_post: f64 = post_coop.iter()
            .map(|s| body_readout(s, &w_out, b_out)).sum::<f64>() / 500.0;
        eprintln!("  seed={}: pre C={:.4} D={:.4} | post C={:.4}",
            seed_idx, mean_c, mean_d, mean_c_post);

        // Phase 3: Time series
        let opp_coop = vec![1.0; n_ts];
        for &alpha in &ts_alphas {
            let mut esn_eval = esn.clone_esn();
            let mut rng_eval = Rng::new(seed + 5000 + (alpha * 100.0) as u64);
            warmup_cooperative(&mut esn_eval, n_warmup, sigma_xi, &mut rng_eval);

            let (states, body_outs, actions) = run_coupled(
                &mut esn_eval, alpha, &w_out, b_out,
                &opp_coop, sigma_xi, &mut rng_eval,
            );
            for t in 0..n_ts {
                let norm: f64 = states[t].iter().map(|x| x * x).sum::<f64>().sqrt();
                println!("coupled,{:.2},{},{},{:.6},{:.6},{:.6}",
                    alpha, seed_idx, t, body_outs[t], actions[t], norm);
            }
        }

        // AllC control
        {
            let mut esn_allc = esn.clone_esn();
            let mut rng_allc = Rng::new(seed + 6000);
            warmup_cooperative(&mut esn_allc, n_warmup, sigma_xi, &mut rng_allc);
            for t in 0..n_ts {
                let body_out = body_readout(&esn_allc.state, &w_out, b_out);
                let norm: f64 = esn_allc.state.iter().map(|x| x * x).sum::<f64>().sqrt();
                esn_allc.step(&vec![1.0, 1.0], sigma_xi, &mut rng_allc);
                println!("allc,1.00,{},{},{:.6},{:.6},{:.6}",
                    seed_idx, t, body_out, 1.0, norm);
            }
        }
    }

    // === Part 2: KL landscape with noisy cooperative opponent ===
    eprintln!("\n=== Part 2: KL landscape (noisy opponent, eps={}) ===", epsilon);
    println!("---KL_LANDSCAPE---");
    println!("alpha,seed,kl_noisy,kl_pure,mean_body_out,mean_action,mean_payoff,action_var");

    for seed_idx in 0..n_seeds {
        let seed = 200 + seed_idx as u64 * 1000;
        let mut rng = Rng::new(seed);
        let mut esn = ESN::new(d, n_in, rho, &mut rng);

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
            &coop_states, &defect_states, target_c, target_d, lambda_reg,
        );

        // Phase 2: Habituation
        for _ in 0..n_habit {
            let mut fast_states = Vec::with_capacity(n_fast);
            for _ in 0..n_fast {
                esn.step(&coop_input, sigma_xi, &mut rng);
                fast_states.push(esn.state.clone());
            }
            esn.oja_update(&fast_states, beta, alpha_oja, rho_min, rho_max);
        }

        // Reference: purely cooperative driving (habituated distribution)
        let mut esn_ref = esn.clone_esn();
        let mut rng_ref = Rng::new(seed + 4000);
        warmup_cooperative(&mut esn_ref, n_warmup, sigma_xi, &mut rng_ref);
        let pure_opp = vec![1.0; kl_samples];
        // For reference, use α=1 (body governance in pure cooperation)
        let (ref_states, _, _) = run_coupled(
            &mut esn_ref, 1.0, &w_out, b_out,
            &pure_opp, sigma_xi, &mut rng_ref,
        );

        // Sweep α with noisy opponent
        for &alpha in &alphas_fine {
            let mut esn_eval = esn.clone_esn();
            let mut rng_eval = Rng::new(seed + 7000 + (alpha * 1000.0) as u64);
            warmup_cooperative(&mut esn_eval, n_warmup, sigma_xi, &mut rng_eval);

            // Generate noisy opponent schedule (same noise for fair comparison?
            // No — each α gets its own RNG fork, but same seed structure)
            let mut rng_opp = Rng::new(seed + 8000); // same opponent noise across α!
            let noisy_opp = noisy_coop_schedule(kl_samples, epsilon, &mut rng_opp);

            let (states_noisy, body_outs, actions) = run_coupled(
                &mut esn_eval, alpha, &w_out, b_out,
                &noisy_opp, sigma_xi, &mut rng_eval,
            );

            // KL(noisy states || pure cooperative states)
            let kl_noisy = kl_knn(&states_noisy, &ref_states, 5).max(0.0);

            // Also measure KL with pure cooperative (should be near 0 for all α)
            let mut esn_pure = esn.clone_esn();
            let mut rng_pure = Rng::new(seed + 9000 + (alpha * 1000.0) as u64);
            warmup_cooperative(&mut esn_pure, n_warmup, sigma_xi, &mut rng_pure);
            let (states_pure, _, _) = run_coupled(
                &mut esn_pure, alpha, &w_out, b_out,
                &pure_opp, sigma_xi, &mut rng_pure,
            );
            let kl_pure = kl_knn(&states_pure, &ref_states, 5).max(0.0);

            let mean_bo: f64 = body_outs.iter().sum::<f64>() / kl_samples as f64;
            let mean_act: f64 = actions.iter().sum::<f64>() / kl_samples as f64;
            let mean_pay: f64 = actions.iter().zip(noisy_opp.iter())
                .map(|(&a, &o)| payoff(a, o, r, s_pay, t_pay, p_pay))
                .sum::<f64>() / kl_samples as f64;
            let act_var: f64 = {
                let m = mean_act;
                actions.iter().map(|&a| (a - m) * (a - m)).sum::<f64>() / kl_samples as f64
            };

            println!("{:.4},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.8}",
                alpha, seed_idx, kl_noisy, kl_pure, mean_bo, mean_act, mean_pay, act_var);
        }
        eprintln!("  seed {} done", seed_idx);
    }
    eprintln!("Done.");
}
