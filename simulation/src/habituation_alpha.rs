// ==========================================================================
// habituation_alpha.rs — Habituation dynamics under different α
//
// Measures KL(q_{α,noisy} || q_{hab}) during habituation.
// Shows how habituation improves noise resilience, and how this varies by α.
//
// Output CSV: mode,alpha,seed,H,kl_noisy,mean_body_out,rho_current,action_var
// ==========================================================================

mod esn;
use esn::*;

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

    let n_burnin = 300;
    let n_train = 1500;
    let n_warmup = 500;
    let n_measure = 1500;
    let target_c = 0.95;
    let target_d = 0.05;
    let lambda_reg = 0.001;
    let epsilon = 0.1;

    let h_max = 300;
    let measure_interval = 15;

    let alphas = [0.0, 0.5, 1.0, -1.0]; // -1.0 signals dynamic sentinel
    let n_seeds = 20;

    let coop_input = vec![1.0, 1.0];
    let defect_input = vec![0.0, 0.0];

    println!("mode,alpha,seed,H,kl_noisy,mean_body_out,rho_current,action_var");

    for seed_idx in 0..n_seeds {
        let seed = 500 + seed_idx as u64 * 1000;

        // Train readout on FRESH reservoir
        let mut rng_dev = Rng::new(seed);
        let esn_dev = ESN::new(d, n_in, rho, &mut rng_dev);

        let mut esn_c = esn_dev.clone_esn();
        let mut rng_c = Rng::new(seed + 2000);
        let coop_states = drive_and_collect(
            &mut esn_c, &coop_input, n_burnin, n_train, sigma_xi, &mut rng_c,
        );
        let mut esn_d = esn_dev.clone_esn();
        let mut rng_d = Rng::new(seed + 3000);
        let defect_states = drive_and_collect(
            &mut esn_d, &defect_input, n_burnin, n_train, sigma_xi, &mut rng_d,
        );
        let (w_out, b_out) = train_readout_ridge(
            &coop_states, &defect_states, target_c, target_d, lambda_reg,
        );

        // Same noisy schedule for all measurements within a seed
        let mut rng_opp = Rng::new(seed + 8000);
        let noisy_opp = noisy_coop_schedule(n_measure, epsilon, &mut rng_opp);

        // === Coupled habituation for each α ===
        for &alpha in &alphas {
            let is_dynamic = alpha < 0.0;
            let effective_alpha_init = if is_dynamic { 0.85 } else { alpha };
            let alpha_label = if is_dynamic { -1.0 } else { alpha };

            let mut rng = Rng::new(seed);
            let mut esn = ESN::new(d, n_in, rho, &mut rng);

            for h in 0..=h_max {
                if h % measure_interval == 0 {
                    // Reference: pure cooperation at current habituation level
                    let mut esn_ref = esn.clone_esn();
                    let mut rng_ref = Rng::new(seed + 4000 + h as u64);
                    warmup_cooperative(&mut esn_ref, n_warmup, sigma_xi, &mut rng_ref);
                    let pure_opp = vec![1.0; n_measure];
                    let mut ref_states = Vec::with_capacity(n_measure);
                    for t in 0..n_measure {
                        let a_body = body_readout(&esn_ref.state, &w_out, b_out);
                        let action = a_body;
                        let input = vec![action, pure_opp[t]];
                        esn_ref.step(&input, sigma_xi, &mut rng_ref);
                        ref_states.push(esn_ref.state.clone());
                    }

                    // Noisy coupled dynamics at current α (or dynamic sentinel)
                    let mut esn_eval = esn.clone_esn();
                    let mut rng_eval = Rng::new(seed + 5000 + h as u64);
                    warmup_cooperative(&mut esn_eval, n_warmup, sigma_xi, &mut rng_eval);

                    let mut sentinel_opt = if is_dynamic {
                        let mut s = Sentinel::default_sentinel(d);
                        // Init baselines
                        let mut esn_bl = esn.clone_esn();
                        let mut rng_bl = Rng::new(seed + 5500 + h as u64);
                        let inp_bl = vec![1.0, 1.0];
                        let n_bl = 200;
                        let mut x_sum = zeros_vec(d);
                        let mut a_sum = 0.0;
                        for _ in 0..n_bl {
                            esn_bl.step(&inp_bl, sigma_xi, &mut rng_bl);
                            let a_b = body_readout(&esn_bl.state, &w_out, b_out);
                            for i in 0..d { x_sum[i] += esn_bl.state[i]; }
                            a_sum += a_b;
                        }
                        let x_mean: Vec1D = x_sum.iter().map(|v| v / n_bl as f64).collect();
                        s.set_baselines(x_mean, a_sum / n_bl as f64);
                        Some(s)
                    } else {
                        None
                    };

                    let mut q_states = Vec::with_capacity(n_measure);
                    let mut actions = Vec::with_capacity(n_measure);
                    let mut last_opp = 1.0;
                    for t in 0..n_measure {
                        let a_body = body_readout(&esn_eval.state, &w_out, b_out);
                        let a_cog = last_opp;
                        let cur_alpha = if let Some(ref mut s) = sentinel_opt {
                            s.update(&esn_eval.state, a_body, a_cog);
                            s.alpha
                        } else {
                            alpha
                        };
                        let action = cur_alpha * a_body + (1.0 - cur_alpha) * a_cog;
                        let input = vec![action, noisy_opp[t]];
                        esn_eval.step(&input, sigma_xi, &mut rng_eval);
                        q_states.push(esn_eval.state.clone());
                        actions.push(action);
                        last_opp = noisy_opp[t];
                    }

                    let kl = kl_knn(&q_states, &ref_states, 5).max(0.0);
                    let mean_bo: f64 = q_states.iter()
                        .map(|s| body_readout(s, &w_out, b_out))
                        .sum::<f64>() / n_measure as f64;
                    let mean_act: f64 = actions.iter().sum::<f64>() / n_measure as f64;
                    let act_var: f64 = {
                        let m = mean_act;
                        actions.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / n_measure as f64
                    };

                    let mut rng_rho = Rng::new(42);
                    let rho_cur = spectral_radius(&esn.w_res, &mut rng_rho);

                    println!("coupled,{:.1},{},{},{:.6},{:.6},{:.4},{:.8}",
                        alpha_label, seed_idx, h, kl, mean_bo, rho_cur, act_var);
                }

                // Habituation step (use effective alpha for habituation dynamics)
                if h < h_max {
                    let mut fast_states = Vec::with_capacity(n_fast);
                    let mut last_opp = 1.0;
                    for _ in 0..n_fast {
                        let a_body = body_readout(&esn.state, &w_out, b_out);
                        let a_cog = last_opp;
                        let action = effective_alpha_init * a_body + (1.0 - effective_alpha_init) * a_cog;
                        let input = vec![action, 1.0];
                        esn.step(&input, sigma_xi, &mut rng);
                        fast_states.push(esn.state.clone());
                        last_opp = 1.0;
                    }
                    esn.oja_update(&fast_states, beta, alpha_oja, rho_min, rho_max);
                }
            }
        }

        eprintln!("seed {} done", seed_idx);
    }
    eprintln!("Done.");
}
