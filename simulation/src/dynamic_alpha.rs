// ==========================================================================
// dynamic_alpha.rs — Dynamic sentinel response & parameter sensitivity
//
// Experiment 6: Multi-phase opponent schedule tests the body-driven sentinel
//   Cooperative(500) → Defect(50) → Coop(500) → Noisy(200,ε=0.3) → Coop(500)
//
// Experiment 7: Parameter sensitivity sweep (Part 2)
//
// Output (two CSV sections separated by ---PARAM_SENSITIVITY---):
//   Part 1: agent,seed,t,body_output,a_cog,alpha_t,d_state,d_output,d_disagree,d_total,action,opp_action,payoff
//   Part 2: param,value,seed,mean_payoff,mean_alpha,mean_d_total,action_var
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

/// Build multi-phase opponent schedule
fn build_schedule(rng: &mut Rng) -> Vec<f64> {
    let mut s = Vec::new();
    // Phase 1: cooperative (500)
    for _ in 0..500 { s.push(1.0); }
    // Phase 2: defection (50)
    for _ in 0..50 { s.push(0.0); }
    // Phase 3: cooperative (500)
    for _ in 0..500 { s.push(1.0); }
    // Phase 4: noisy cooperative (200, epsilon=0.3)
    for _ in 0..200 {
        if rng.next_f64() < 0.3 { s.push(0.0); } else { s.push(1.0); }
    }
    // Phase 5: cooperative (500)
    for _ in 0..500 { s.push(1.0); }
    s
}

/// Run coupled dynamics with dynamic sentinel.
/// Returns (body_outs, a_cogs, alpha_trace, d_states, d_outputs, d_disagrees, d_totals, actions)
fn run_coupled_dynamic(
    esn: &mut ESN,
    sentinel: &mut Sentinel,
    w_out: &Vec1D,
    b_out: f64,
    opp_actions: &[f64],
    sigma_xi: f64,
    rng: &mut Rng,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = opp_actions.len();
    let mut body_outs = Vec::with_capacity(n);
    let mut a_cogs = Vec::with_capacity(n);
    let mut alpha_trace = Vec::with_capacity(n);
    let mut d_states = Vec::with_capacity(n);
    let mut d_outputs = Vec::with_capacity(n);
    let mut d_disagrees = Vec::with_capacity(n);
    let mut d_totals = Vec::with_capacity(n);
    let mut actions = Vec::with_capacity(n);
    let mut last_opp = opp_actions[0];

    for t in 0..n {
        let a_opp = opp_actions[t];
        let a_body = body_readout(&esn.state, w_out, b_out);
        let a_cog = last_opp; // TfT

        // Sentinel observes body state and outputs, then adjusts alpha
        sentinel.update(&esn.state, a_body, a_cog);
        let alpha = sentinel.alpha;

        let action = alpha * a_body + (1.0 - alpha) * a_cog;
        let input = vec![action, a_opp];
        esn.step(&input, sigma_xi, rng);

        body_outs.push(a_body);
        a_cogs.push(a_cog);
        alpha_trace.push(alpha);
        d_states.push(sentinel.last_d_state);
        d_outputs.push(sentinel.last_d_output);
        d_disagrees.push(sentinel.last_d_disagree);
        d_totals.push(sentinel.last_d_total);
        actions.push(action);
        last_opp = a_opp;
    }
    (body_outs, a_cogs, alpha_trace, d_states, d_outputs, d_disagrees, d_totals, actions)
}

/// Run coupled dynamics with static alpha.
fn run_coupled_static(
    esn: &mut ESN,
    alpha: f64,
    w_out: &Vec1D,
    b_out: f64,
    opp_actions: &[f64],
    sigma_xi: f64,
    rng: &mut Rng,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = opp_actions.len();
    let mut body_outs = Vec::with_capacity(n);
    let mut actions = Vec::with_capacity(n);
    let mut a_cogs = Vec::with_capacity(n);
    let mut last_opp = opp_actions[0];

    for t in 0..n {
        let a_opp = opp_actions[t];
        let a_body = body_readout(&esn.state, w_out, b_out);
        let a_cog = last_opp;
        let action = alpha * a_body + (1.0 - alpha) * a_cog;
        let input = vec![action, a_opp];
        esn.step(&input, sigma_xi, rng);
        body_outs.push(a_body);
        actions.push(action);
        a_cogs.push(a_cog);
        last_opp = a_opp;
    }
    (body_outs, actions, a_cogs)
}

/// Initialize sentinel baselines from warmup phase
fn init_sentinel_baselines(
    esn: &ESN,
    sentinel: &mut Sentinel,
    w_out: &Vec1D,
    b_out: f64,
    sigma_xi: f64,
    n_steps: usize,
    rng: &mut Rng,
) {
    let mut esn_tmp = esn.clone_esn();
    let inp = vec![1.0, 1.0];
    let mut x_sum = zeros_vec(esn.d);
    let mut a_sum = 0.0;
    for _ in 0..n_steps {
        esn_tmp.step(&inp, sigma_xi, rng);
        let a_body = body_readout(&esn_tmp.state, w_out, b_out);
        for i in 0..esn.d {
            x_sum[i] += esn_tmp.state[i];
        }
        a_sum += a_body;
    }
    let n = n_steps as f64;
    let x_mean: Vec1D = x_sum.iter().map(|v| v / n).collect();
    sentinel.set_baselines(x_mean, a_sum / n);
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

    let n_burnin = 500;
    let n_train = 2000;
    let n_warmup = 500;
    let target_c = 0.95;
    let target_d = 0.05;
    let lambda_reg = 0.001;

    let coop_input = vec![1.0, 1.0];
    let defect_input = vec![0.0, 0.0];
    let n_seeds = 20;

    // ====================================================================
    // Part 1: Multi-phase opponent schedule
    // ====================================================================
    eprintln!("=== Part 1: Dynamic sentinel multi-phase response ===");
    println!("agent,seed,t,body_output,a_cog,alpha_t,d_state,d_output,d_disagree,d_total,action,opp_action,payoff");

    let static_alphas = [("alpha0", 0.0), ("alpha07", 0.7), ("alpha085", 0.85), ("alpha1", 1.0)];

    for seed_idx in 0..n_seeds {
        let seed = 500 + seed_idx as u64 * 1000;
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

        // Phase 3: Multi-phase perturbation test

        // Generate opponent schedule (same for all agents in this seed)
        let mut rng_sched = Rng::new(seed + 4000);
        let opp_schedule = build_schedule(&mut rng_sched);
        let t_total = opp_schedule.len();

        // Dynamic sentinel agent
        {
            let mut esn_eval = esn.clone_esn();
            let mut rng_eval = Rng::new(seed + 5000);
            warmup_cooperative(&mut esn_eval, n_warmup, sigma_xi, &mut rng_eval);

            let mut sentinel = Sentinel::default_sentinel(d);
            let mut rng_bl = Rng::new(seed + 5500);
            init_sentinel_baselines(&esn, &mut sentinel, &w_out, b_out, sigma_xi, 200, &mut rng_bl);

            let (body_outs, a_cogs, alpha_trace, d_states, d_outputs, d_disagrees, d_totals, actions) =
                run_coupled_dynamic(
                    &mut esn_eval, &mut sentinel, &w_out, b_out,
                    &opp_schedule, sigma_xi, &mut rng_eval,
                );
            for t in 0..t_total {
                let pay = payoff(actions[t], opp_schedule[t]);
                println!("dynamic,{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.1},{:.6}",
                    seed_idx, t, body_outs[t], a_cogs[t], alpha_trace[t],
                    d_states[t], d_outputs[t], d_disagrees[t], d_totals[t],
                    actions[t], opp_schedule[t], pay);
            }
        }

        // Static alpha agents
        for &(name, alpha) in &static_alphas {
            let mut esn_eval = esn.clone_esn();
            let mut rng_eval = Rng::new(seed + 6000 + (alpha * 1000.0) as u64);
            warmup_cooperative(&mut esn_eval, n_warmup, sigma_xi, &mut rng_eval);

            let (body_outs, actions, a_cogs) = run_coupled_static(
                &mut esn_eval, alpha, &w_out, b_out,
                &opp_schedule, sigma_xi, &mut rng_eval,
            );
            for t in 0..t_total {
                let pay = payoff(actions[t], opp_schedule[t]);
                println!("{},{},{},{:.6},{:.6},{:.6},0,0,0,0,{:.6},{:.1},{:.6}",
                    name, seed_idx, t, body_outs[t], a_cogs[t], alpha,
                    actions[t], opp_schedule[t], pay);
            }
        }
        eprintln!("  seed {} done", seed_idx);
    }

    // ====================================================================
    // Part 2: Parameter sensitivity
    // ====================================================================
    eprintln!("\n=== Part 2: Parameter sensitivity ===");
    println!("---PARAM_SENSITIVITY---");
    println!("param,value,seed,mean_payoff,mean_alpha,mean_d_total,action_var");

    let epsilon_sens = 0.1;
    let kl_samples = 2000;
    let n_seeds_sens = 20;

    // Parameter sweeps
    let alpha0_vals = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95];
    let eta_up_vals = [0.01, 0.02, 0.05, 0.1, 0.2];
    let eta_down_vals = [0.1, 0.3, 0.5, 0.8, 1.0];
    let theta_vals = [0.0, 0.05, 0.1, 0.2, 0.3];

    for seed_idx in 0..n_seeds_sens {
        let seed = 700 + seed_idx as u64 * 1000;
        let mut rng = Rng::new(seed);
        let mut esn = ESN::new(d, n_in, rho, &mut rng);

        // Development
        let mut esn_c = esn.clone_esn();
        let mut rng_c = Rng::new(seed + 2000);
        let coop_states = drive_and_collect(
            &mut esn_c, &coop_input, n_burnin, n_train, sigma_xi, &mut rng_c,
        );
        let mut esn_d2 = esn.clone_esn();
        let mut rng_d2 = Rng::new(seed + 3000);
        let defect_states = drive_and_collect(
            &mut esn_d2, &defect_input, n_burnin, n_train, sigma_xi, &mut rng_d2,
        );
        let (w_out, b_out) = train_readout_ridge(
            &coop_states, &defect_states, target_c, target_d, lambda_reg,
        );

        // Habituation
        for _ in 0..n_habit {
            let mut fast_states = Vec::with_capacity(n_fast);
            for _ in 0..n_fast {
                esn.step(&coop_input, sigma_xi, &mut rng);
                fast_states.push(esn.state.clone());
            }
            esn.oja_update(&fast_states, beta, alpha_oja, rho_min, rho_max);
        }

        // Helper: run one sensitivity trial
        let run_trial = |esn_base: &ESN, sentinel_params: (f64, f64, f64, f64, f64), seed_offset: u64| -> (f64, f64, f64, f64) {
            let mut esn_eval = esn_base.clone_esn();
            let mut rng_eval = Rng::new(seed + seed_offset);
            warmup_cooperative(&mut esn_eval, n_warmup, sigma_xi, &mut rng_eval);

            let (a0, eu, ed, th, rr) = sentinel_params;
            let mut sentinel = Sentinel::new(
                d, a0, eu, ed, th, 0.05, 0.3, 0.3, 0.4, rr, 0.02
            );
            let mut rng_bl = Rng::new(seed + seed_offset + 100);
            init_sentinel_baselines(esn_base, &mut sentinel, &w_out, b_out, sigma_xi, 200, &mut rng_bl);

            let mut rng_opp = Rng::new(seed + 8000);
            let noisy_opp: Vec<f64> = (0..kl_samples).map(|_| {
                if rng_opp.next_f64() < epsilon_sens { 0.0 } else { 1.0 }
            }).collect();

            let (_, _, alpha_trace, _, _, _, d_totals, actions) =
                run_coupled_dynamic(
                    &mut esn_eval, &mut sentinel, &w_out, b_out,
                    &noisy_opp, sigma_xi, &mut rng_eval,
                );

            let mean_pay: f64 = actions.iter().zip(&noisy_opp)
                .map(|(&a, &o)| payoff(a, o)).sum::<f64>() / kl_samples as f64;
            let mean_alpha: f64 = alpha_trace.iter().sum::<f64>() / kl_samples as f64;
            let mean_d: f64 = d_totals.iter().sum::<f64>() / kl_samples as f64;
            let mean_act: f64 = actions.iter().sum::<f64>() / kl_samples as f64;
            let act_var: f64 = actions.iter().map(|&a| (a - mean_act) * (a - mean_act))
                .sum::<f64>() / kl_samples as f64;

            (mean_pay, mean_alpha, mean_d, act_var)
        };

        // Sweep alpha_0
        for &a0 in &alpha0_vals {
            let (mp, ma, md, av) = run_trial(&esn, (a0, 0.05, 0.5, 0.1, 1.0), 10000 + (a0 * 100.0) as u64);
            println!("alpha_0,{:.2},{},{:.6},{:.6},{:.6},{:.8}", a0, seed_idx, mp, ma, md, av);
        }
        // Sweep eta_up
        for &eu in &eta_up_vals {
            let (mp, ma, md, av) = run_trial(&esn, (0.85, eu, 0.5, 0.1, 1.0), 20000 + (eu * 1000.0) as u64);
            println!("eta_up,{:.3},{},{:.6},{:.6},{:.6},{:.8}", eu, seed_idx, mp, ma, md, av);
        }
        // Sweep eta_down
        for &ed in &eta_down_vals {
            let (mp, ma, md, av) = run_trial(&esn, (0.85, 0.05, ed, 0.1, 1.0), 30000 + (ed * 100.0) as u64);
            println!("eta_down,{:.1},{},{:.6},{:.6},{:.6},{:.8}", ed, seed_idx, mp, ma, md, av);
        }
        // Sweep theta
        for &th in &theta_vals {
            let (mp, ma, md, av) = run_trial(&esn, (0.85, 0.05, 0.5, th, 1.0), 40000 + (th * 1000.0) as u64);
            println!("theta,{:.2},{},{:.6},{:.6},{:.6},{:.8}", th, seed_idx, mp, ma, md, av);
        }
        eprintln!("  sensitivity seed {} done", seed_idx);
    }
    // ====================================================================
    // Part 3: Defection block length sweep (phase transition analysis)
    // ====================================================================
    eprintln!("\n=== Part 3: Defection block length sweep ===");
    println!("---PHASE_TRANSITION_ENV---");
    println!("defect_length,seed,detection_time,alpha_min_reached,recovery_time,payoff_during_defect,payoff_total");

    let defect_lengths: Vec<usize> = vec![10, 20, 50, 100, 200, 500];
    let n_seeds_pt = 20;

    for &defect_len in &defect_lengths {
        for seed_idx in 0..n_seeds_pt {
            let seed = 1100 + seed_idx as u64 * 1000 + defect_len as u64 * 10;
            let mut rng = Rng::new(seed);
            let mut esn = ESN::new(d, n_in, rho, &mut rng);

            // Development
            let mut esn_c = esn.clone_esn();
            let mut rng_c = Rng::new(seed + 2000);
            let coop_states = drive_and_collect(
                &mut esn_c, &coop_input, n_burnin, n_train, sigma_xi, &mut rng_c,
            );
            let mut esn_d2 = esn.clone_esn();
            let mut rng_d2 = Rng::new(seed + 3000);
            let defect_states = drive_and_collect(
                &mut esn_d2, &defect_input, n_burnin, n_train, sigma_xi, &mut rng_d2,
            );
            let (w_out, b_out) = train_readout_ridge(
                &coop_states, &defect_states, target_c, target_d, lambda_reg,
            );

            // Habituation
            for _ in 0..n_habit {
                let mut fast_states = Vec::with_capacity(n_fast);
                for _ in 0..n_fast {
                    esn.step(&coop_input, sigma_xi, &mut rng);
                    fast_states.push(esn.state.clone());
                }
                esn.oja_update(&fast_states, beta, alpha_oja, rho_min, rho_max);
            }

            // Build schedule: Coop(500) → Defect(defect_len) → Coop(500)
            let mut opp_schedule = Vec::new();
            for _ in 0..500 { opp_schedule.push(1.0); }
            for _ in 0..defect_len { opp_schedule.push(0.0); }
            for _ in 0..500 { opp_schedule.push(1.0); }
            let t_total = opp_schedule.len();

            // Run dynamic sentinel
            let mut esn_eval = esn.clone_esn();
            let mut rng_eval = Rng::new(seed + 5000);
            warmup_cooperative(&mut esn_eval, n_warmup, sigma_xi, &mut rng_eval);

            let mut sentinel = Sentinel::default_sentinel(d);
            let mut rng_bl = Rng::new(seed + 5500);
            init_sentinel_baselines(&esn, &mut sentinel, &w_out, b_out, sigma_xi, 200, &mut rng_bl);

            let (_, _, alpha_trace, _, _, _, _, actions) =
                run_coupled_dynamic(
                    &mut esn_eval, &mut sentinel, &w_out, b_out,
                    &opp_schedule, sigma_xi, &mut rng_eval,
                );

            // Compute metrics
            let defect_start = 500;
            let defect_end = 500 + defect_len;

            // Detection time: first t >= defect_start where alpha < alpha_0 - 0.1
            let detection_time = {
                let mut dt = defect_len; // default: not detected within block
                for t in defect_start..defect_end {
                    if alpha_trace[t] < 0.75 {
                        dt = t - defect_start;
                        break;
                    }
                }
                dt
            };

            // Minimum alpha reached during defection block
            let alpha_min_reached = alpha_trace[defect_start..defect_end]
                .iter().cloned().fold(f64::INFINITY, f64::min);

            // Recovery time: first t >= defect_end where alpha > 0.7
            let recovery_time = {
                let mut rt = t_total - defect_end; // default: not recovered
                for t in defect_end..t_total {
                    if alpha_trace[t] > 0.7 {
                        rt = t - defect_end;
                        break;
                    }
                }
                rt
            };

            // Payoff during defection block
            let payoff_during: f64 = (defect_start..defect_end)
                .map(|t| payoff(actions[t], opp_schedule[t]))
                .sum::<f64>() / defect_len as f64;

            // Total payoff
            let payoff_total: f64 = (0..t_total)
                .map(|t| payoff(actions[t], opp_schedule[t]))
                .sum::<f64>();

            println!("{},{},{},{:.6},{},{:.6},{:.4}",
                defect_len, seed_idx, detection_time, alpha_min_reached,
                recovery_time, payoff_during, payoff_total);
        }
        eprintln!("  defect_len={} done", defect_len);
    }

    eprintln!("Done.");
}
