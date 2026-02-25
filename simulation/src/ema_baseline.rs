// ==========================================================================
// ema_baseline.rs — EMA-filtered TfT baseline comparison
//
// Compares the reservoir-governed agent against a simple EMA-filtered TfT
// strategy to disentangle temporal smoothing from implicit inference.
//
// EMA-TfT: a_ema(t) = gamma * a_ema(t-1) + (1-gamma) * opponent(t-1)
//
// Measures: action variance, KL divergence, mean payoff, anomaly response
//
// Output CSV:
//   agent,gamma,seed,action_var,kl_noisy,mean_payoff,mean_action,
//   perturbation_depth,recovery_time
// ==========================================================================

mod esn;
use esn::*;

fn payoff(a_i: f64, a_j: f64, r: f64, s: f64, t: f64, p: f64) -> f64 {
    r * a_i * a_j + s * a_i * (1.0 - a_j) + t * (1.0 - a_i) * a_j + p * (1.0 - a_i) * (1.0 - a_j)
}

fn noisy_coop_schedule(n: usize, epsilon: f64, rng: &mut Rng) -> Vec<f64> {
    (0..n).map(|_| {
        if rng.next_f64() < epsilon { 0.0 } else { 1.0 }
    }).collect()
}

/// Perturbation schedule: cooperate, then defect block, then cooperate
fn perturbation_schedule(n_pre: usize, n_defect: usize, n_post: usize) -> Vec<f64> {
    let mut s = Vec::with_capacity(n_pre + n_defect + n_post);
    s.extend(vec![1.0; n_pre]);
    s.extend(vec![0.0; n_defect]);
    s.extend(vec![1.0; n_post]);
    s
}

/// Run EMA-filtered TfT agent
/// Paper Eq.: a_ema(t) = gamma * a_ema(t-1) + (1-gamma) * a_opp(t-1)
fn run_ema_tft(
    opp_actions: &[f64],
    gamma: f64,
) -> Vec<f64> {
    let n = opp_actions.len();
    let mut actions = Vec::with_capacity(n);
    let mut a_ema = 1.0; // start cooperative
    let mut last_opp = 1.0; // assume cooperative initial opponent

    for t in 0..n {
        // Update EMA with opponent's *previous* action (TfT timing)
        a_ema = gamma * a_ema + (1.0 - gamma) * last_opp;
        actions.push(a_ema);
        last_opp = opp_actions[t];
    }
    actions
}

/// Run reservoir-governed agent (α=1)
fn run_reservoir(
    esn: &mut ESN,
    w_out: &Vec1D,
    b_out: f64,
    opp_actions: &[f64],
    sigma_xi: f64,
    rng: &mut Rng,
) -> (Vec<Vec1D>, Vec<f64>) {
    let n = opp_actions.len();
    let mut states = Vec::with_capacity(n);
    let mut actions = Vec::with_capacity(n);

    for t in 0..n {
        let a_body = body_readout(&esn.state, w_out, b_out);
        let input = vec![a_body, opp_actions[t]];
        esn.step(&input, sigma_xi, rng);
        states.push(esn.state.clone());
        actions.push(a_body);
    }
    (states, actions)
}

fn warmup_cooperative(esn: &mut ESN, n_steps: usize, sigma_xi: f64, rng: &mut Rng) {
    let inp = vec![1.0, 1.0];
    for _ in 0..n_steps {
        esn.step(&inp, sigma_xi, rng);
    }
}

fn action_variance(actions: &[f64]) -> f64 {
    let n = actions.len() as f64;
    let mean = actions.iter().sum::<f64>() / n;
    actions.iter().map(|&a| (a - mean) * (a - mean)).sum::<f64>() / n
}

/// Measure perturbation depth (how low does action go) and recovery time
fn perturbation_metrics(actions: &[f64], n_pre: usize, n_defect: usize) -> (f64, usize) {
    let perturb_start = n_pre;
    let perturb_end = n_pre + n_defect;

    // Depth: minimum action during perturbation
    let depth = actions[perturb_start..perturb_end]
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);

    // Recovery: steps after perturbation ends to reach 0.95 of pre-perturbation mean
    let pre_mean: f64 = actions[..perturb_start].iter().sum::<f64>() / perturb_start as f64;
    let threshold = 0.95 * pre_mean;
    let mut recovery = 0usize;
    for (i, &a) in actions[perturb_end..].iter().enumerate() {
        if a >= threshold {
            recovery = i;
            break;
        }
        recovery = i + 1;
    }
    (depth, recovery)
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

    let n_eval = 2000;
    let epsilon = 0.1;

    let r = 3.0; let s_pay = 0.0; let t_pay = 5.0; let p_pay = 1.0;

    let n_seeds = 20;
    let gammas = [0.0, 0.5, 0.9, 0.95, 0.99]; // EMA smoothing params

    let coop_input = vec![1.0, 1.0];
    let defect_input = vec![0.0, 0.0];

    // Perturbation schedule params
    let n_pre = 200;
    let n_defect_block = 100;
    let n_post = 200;

    // === Part 1: Variance and KL comparison ===
    eprintln!("=== EMA Baseline: Variance & KL comparison ===");
    println!("agent,gamma,seed,action_var,kl_noisy,mean_payoff,mean_action,perturbation_depth,recovery_time");

    for seed_idx in 0..n_seeds {
        let seed = 300 + seed_idx as u64 * 1000;
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

        // Reference distribution (α=1, pure cooperation)
        let mut esn_ref = esn.clone_esn();
        let mut rng_ref = Rng::new(seed + 4000);
        warmup_cooperative(&mut esn_ref, n_warmup, sigma_xi, &mut rng_ref);
        let pure_opp = vec![1.0; n_eval];
        let (ref_states, _) = run_reservoir(
            &mut esn_ref, &w_out, b_out, &pure_opp, sigma_xi, &mut rng_ref,
        );

        // Same noisy opponent for all agents
        let mut rng_opp = Rng::new(seed + 8000);
        let noisy_opp = noisy_coop_schedule(n_eval, epsilon, &mut rng_opp);

        // Same perturbation schedule for all agents
        let perturb_opp = perturbation_schedule(n_pre, n_defect_block, n_post);

        // --- Reservoir agent (α=1) ---
        {
            let mut esn_eval = esn.clone_esn();
            let mut rng_eval = Rng::new(seed + 5000);
            warmup_cooperative(&mut esn_eval, n_warmup, sigma_xi, &mut rng_eval);

            let (states_noisy, actions_noisy) = run_reservoir(
                &mut esn_eval, &w_out, b_out, &noisy_opp, sigma_xi, &mut rng_eval,
            );
            let var_res = action_variance(&actions_noisy);
            let kl_res = kl_knn(&states_noisy, &ref_states, 5).max(0.0);
            let mean_pay_res: f64 = actions_noisy.iter().zip(noisy_opp.iter())
                .map(|(&a, &o)| payoff(a, o, r, s_pay, t_pay, p_pay))
                .sum::<f64>() / n_eval as f64;
            let mean_act_res: f64 = actions_noisy.iter().sum::<f64>() / n_eval as f64;

            // Perturbation response
            let mut esn_pert = esn.clone_esn();
            let mut rng_pert = Rng::new(seed + 6000);
            warmup_cooperative(&mut esn_pert, n_warmup, sigma_xi, &mut rng_pert);
            let (_, actions_pert) = run_reservoir(
                &mut esn_pert, &w_out, b_out, &perturb_opp, sigma_xi, &mut rng_pert,
            );
            let (depth, recovery) = perturbation_metrics(&actions_pert, n_pre, n_defect_block);

            println!("reservoir,1.000,{},{:.8},{:.6},{:.6},{:.6},{:.6},{}",
                seed_idx, var_res, kl_res, mean_pay_res, mean_act_res, depth, recovery);
        }

        // --- Raw TfT (γ=0, equivalent to copying last action) ---
        {
            let actions_tft = run_ema_tft(&noisy_opp, 0.0);
            let var_tft = action_variance(&actions_tft);
            let mean_pay_tft: f64 = actions_tft.iter().zip(noisy_opp.iter())
                .map(|(&a, &o)| payoff(a, o, r, s_pay, t_pay, p_pay))
                .sum::<f64>() / n_eval as f64;
            let mean_act_tft: f64 = actions_tft.iter().sum::<f64>() / n_eval as f64;

            // Perturbation
            let actions_pert = run_ema_tft(&perturb_opp, 0.0);
            let (depth, recovery) = perturbation_metrics(&actions_pert, n_pre, n_defect_block);

            // No KL for EMA (no reservoir states)
            println!("tft,0.000,{},{:.8},NA,{:.6},{:.6},{:.6},{}",
                seed_idx, var_tft, mean_pay_tft, mean_act_tft, depth, recovery);
        }

        // --- EMA-filtered TfT at various γ ---
        for &gamma in &gammas {
            if gamma == 0.0 { continue; } // already done as raw TfT

            let actions_ema = run_ema_tft(&noisy_opp, gamma);
            let var_ema = action_variance(&actions_ema);
            let mean_pay_ema: f64 = actions_ema.iter().zip(noisy_opp.iter())
                .map(|(&a, &o)| payoff(a, o, r, s_pay, t_pay, p_pay))
                .sum::<f64>() / n_eval as f64;
            let mean_act_ema: f64 = actions_ema.iter().sum::<f64>() / n_eval as f64;

            // Perturbation
            let actions_pert = run_ema_tft(&perturb_opp, gamma);
            let (depth, recovery) = perturbation_metrics(&actions_pert, n_pre, n_defect_block);

            println!("ema_tft,{:.3},{},{:.8},NA,{:.6},{:.6},{:.6},{}",
                gamma, seed_idx, var_ema, mean_pay_ema, mean_act_ema, depth, recovery);
        }

        eprintln!("  seed {} done", seed_idx);
    }
    eprintln!("Done.");
}
