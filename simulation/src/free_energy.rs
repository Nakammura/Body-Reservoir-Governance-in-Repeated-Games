// ==========================================================================
// free_energy.rs — Free energy landscape FE(α, H) with noisy opponent
//
// Uses noisy cooperative opponent (ε=0.1) and measures KL against
// the habituated cooperation distribution (not natural distribution).
// This shows the meaningful α-dependence of complexity cost.
//
// FE(α) = -payoff(α) + λ · KL(q_{α,noisy} || q_hab)
//
// Output CSV: H,alpha,seed,kl,mean_payoff,mean_action,mean_body_out,fe_low,fe_mid,fe_high
// ==========================================================================

mod esn;
use esn::*;

fn payoff(a_i: f64, a_j: f64, r: f64, s: f64, t: f64, p: f64) -> f64 {
    r * a_i * a_j + s * a_i * (1.0 - a_j) + t * (1.0 - a_i) * a_j + p * (1.0 - a_i) * (1.0 - a_j)
}

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

    let n_burnin = 500;
    let n_train = 2000;
    let n_warmup = 500;
    let kl_samples = 2000;
    let target_c = 0.95;
    let target_d = 0.05;
    let lambda_reg = 0.001;
    let epsilon = 0.1;

    let r = 3.0; let s_pay = 0.0; let t_pay = 5.0; let p_pay = 1.0;
    let lambda_low = 1.0;
    let lambda_mid = 3.0;
    let lambda_high = 8.0;

    let h_values = [0_usize, 10, 25, 50, 100, 200];
    let alphas: Vec<f64> = (0..=10).map(|i| i as f64 * 0.1).collect();
    let n_seeds = 20;

    let coop_input = vec![1.0, 1.0];
    let defect_input = vec![0.0, 0.0];

    println!("H,alpha,seed,kl,mean_payoff,mean_action,mean_body_out,fe_low,fe_mid,fe_high");

    for &h in &h_values {
        for seed_idx in 0..n_seeds {
            let seed = 400 + seed_idx as u64 * 1000 + h as u64 * 10;
            let mut rng = Rng::new(seed);
            let mut esn = ESN::new(d, n_in, rho, &mut rng);

            // Phase 1: Development (on fresh reservoir)
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

            // Phase 2: Habituation (Oja only)
            for _ in 0..h {
                let mut fast_states = Vec::with_capacity(n_fast);
                for _ in 0..n_fast {
                    esn.step(&coop_input, sigma_xi, &mut rng);
                    fast_states.push(esn.state.clone());
                }
                esn.oja_update(&fast_states, beta, alpha_oja, rho_min, rho_max);
            }

            // Reference: habituated cooperation distribution (pure [1,1] driving)
            let mut esn_ref = esn.clone_esn();
            let mut rng_ref = Rng::new(seed + 4000);
            warmup_cooperative(&mut esn_ref, n_warmup, sigma_xi, &mut rng_ref);
            let pure_opp = vec![1.0; kl_samples];
            let (ref_states, _, _) = run_coupled(
                &mut esn_ref, 1.0, &w_out, b_out,
                &pure_opp, sigma_xi, &mut rng_ref,
            );

            // Same noisy schedule for all α (fair comparison)
            let mut rng_opp = Rng::new(seed + 8000);
            let noisy_opp = noisy_coop_schedule(kl_samples, epsilon, &mut rng_opp);

            // Sweep α
            for &alpha in &alphas {
                let mut esn_eval = esn.clone_esn();
                let mut rng_eval = Rng::new(seed + 7000 + (alpha * 1000.0) as u64);
                warmup_cooperative(&mut esn_eval, n_warmup, sigma_xi, &mut rng_eval);

                let (states, body_outs, actions) = run_coupled(
                    &mut esn_eval, alpha, &w_out, b_out,
                    &noisy_opp, sigma_xi, &mut rng_eval,
                );

                let kl = kl_knn(&states, &ref_states, 5).max(0.0);
                let mean_pay: f64 = actions.iter().zip(noisy_opp.iter())
                    .map(|(&a, &o)| payoff(a, o, r, s_pay, t_pay, p_pay))
                    .sum::<f64>() / kl_samples as f64;
                let mean_act: f64 = actions.iter().sum::<f64>() / kl_samples as f64;
                let mean_bo: f64 = body_outs.iter().sum::<f64>() / kl_samples as f64;

                let fe_low = -mean_pay + lambda_low * kl;
                let fe_mid = -mean_pay + lambda_mid * kl;
                let fe_high = -mean_pay + lambda_high * kl;

                println!("{},{:.2},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                    h, alpha, seed_idx, kl, mean_pay, mean_act, mean_bo,
                    fe_low, fe_mid, fe_high);
            }
        }
        eprintln!("H={} done", h);
    }
    eprintln!("Done.");
}
