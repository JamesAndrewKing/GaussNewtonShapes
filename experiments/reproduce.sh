#!/bin/bash
## Every training run, benchmark and figure behind the paper. Run from the repository root:
##
##     bash experiments/reproduce.sh            # everything (about 60 GPU-hours on an RTX 5080)
##     bash experiments/reproduce.sh main goldschmidt   # only some sections (names below)
##     DRY=1 bash experiments/reproduce.sh      # list what is not done yet (benchmarks and figures always), without running anything
##
## Runs that already have an evals.json are skipped, so the script can be interrupted and restarted. Budgets are
## seconds of training time (evaluation excluded), so the number of iterations, and with it the results, depend on the
## GPU; run one job at a time, without other GPU load. Seeds fix the initialization and the sampling, but GPU kernels
## are not bitwise deterministic. Each run's settings, including the git commit, are in its config.json.
## The defaults of the tasks (experiments/tasks/) are the settings of the paper; the arguments below are all that changes.
## Our runs: RTX 5080 (16 GB), driver 591.86, Python 3.12.7, torch 2.7.1+cu128, float64.

cd "$(dirname "$0")/.."
R=experiments/results

go() {  ## task opt seed budget exp [more run.py arguments]; the run directory name is built as in run.py
  local t=$1 o=$2 s=$3 b=$4 e=$5; shift 5
  local name=$(python -c "
import json, sys
args, tag, i = {}, '', 0
a = sys.argv[1:]
while i < len(a):
    if a[i] == '--tag': tag = a[i + 1]; i += 2
    elif a[i] == '--task-args':
        i += 1
        while i < len(a) and not a[i].startswith('--'):
            k, v = a[i].split('=', 1); args[k] = json.loads(v.lower()); i += 1
    else: i += 1
print('$o' + tag + ''.join(f'_{k}{v}' for k, v in args.items()) + '_seed$s')" "$@")
  if [ -f $R/$t/$e/$name/evals.json ]; then echo "skip $t/$e/$name"; return; fi
  if [ -n "$DRY" ]; then echo "to run: $t/$e/$name"; return; fi
  mkdir -p $R/$t/$e
  echo "$(date +%H:%M) $t/$e/$name"
  python -m experiments.run --task $t --opt $o --seed $s --budget $b --exp $e "$@" > $R/$t/$e/$name.log 2>&1 || echo "FAILED $t/$e/$name"
}
py() {  ## python, or only print the command with DRY=1
  if [ -n "$DRY" ]; then echo "to run: python $*"; else python "$@"; fi
}
want() { [ ${#SECTIONS[@]} -eq 0 ] || [[ " ${SECTIONS[*]} " == *" $1 "* ]]; }
SECTIONS=("$@")
BUDGET=(catenoid 1200 enneper 1200 cone 1200 rockerarm 600 bunny 600 jeb 1200)
budget() { for ((i = 0; i < ${#BUDGET[@]}; i += 2)); do [ ${BUDGET[i]} = $1 ] && echo ${BUDGET[i+1]}; done; }

## main: Figures 1-8 and the statistics over seeds. Minimal and developable surfaces and INS: 5 seeds.
## JEB: the figure runs start from the pretrained model of the paper (notebooks/results/model-Adam-pretrain.pt), the
## seed runs pretrain their own (pretrained="own"); Figure 7 clips the strain at 1000.
if want main; then
  for s in 0 1 2 3 4; do for t in catenoid enneper cone rockerarm bunny; do for o in gn adam lbfgs; do
    go $t $o $s $(budget $t) main
  done; done; done
  for o in gn adam lbfgs; do
    go jeb $o 0 1200 main
    go jeb $o 0 1200 main --task-args strain_clip=1000
    for s in 0 1 2; do go jeb $o $s 1200 main --task-args 'pretrained="own"'; done
  done
fi

## lbfgs_eikonal: choice of the LBFGS eikonal weights (strong-Wolfe line search, 300 s)
if want lbfgs_eikonal; then
  for t in catenoid enneper cone; do for w in 1 0.1 0.01 0.001 0.0001; do
    go $t lbfgs 0 300 lbfgs_sweep --tag _wolfe_eik$w --task-attrs "eikonal_weights={\"lbfgs\":$w}"
  done; done
fi

## baselines: GN with adaptive damping (lm) and the hybrids Adam -> LBFGS, Adam -> GN, at the main budgets
if want baselines; then
  for s in 0 1 2; do
    for t in catenoid enneper cone rockerarm bunny; do for o in lm adam_lbfgs; do go $t $o $s $(budget $t) main; done; done
    for o in lm adam_lbfgs; do go jeb $o $s 1200 main --task-args 'pretrained="own"'; done
  done
  for t in catenoid enneper cone rockerarm bunny; do go $t adam_gn 0 $(budget $t) main; done
  go jeb adam_gn 0 1200 main --task-args 'pretrained="own"'
fi

## initialization: other starting shapes (catenoid: cylinders of other radii; Enneper: shifted and tilted planes), 300 s, 3 seeds
if want initialization; then
  for s in 0 1 2; do for o in gn adam lbfgs; do
    for r in 0.5 1.0 1.543 1.8; do go catenoid $o $s 300 init_sweep --task-args init_radius=$r pretrain_extent=2.0; done
    go enneper $o $s 300 init_sweep
    go enneper $o $s 300 init_sweep --task-args offset=0.25
    go enneper $o $s 300 init_sweep --task-args offset=0.5
    go enneper $o $s 300 init_sweep --task-args tilt=15.0
  done; done
fi

## goldschmidt: catenoid rings at separations h/R from 0.4 to 0.9, started from the cylinder and from the planes, 300 s
if want goldschmidt; then
  for r in 0.4 0.55 0.648 0.7 0.9; do for i in cylinder planes; do for o in gn adam lbfgs; do
    go catenoid $o 0 300 goldschmidt --task-args h_over_R=$r "init=\"$i\""
  done; done; done
fi

## noise_sparse: noisy points and normals, and a fraction of the points (error on the held-out rest), 600 s
if want noise_sparse; then
  for t in rockerarm bunny; do for o in gn adam; do
    for n in 0.0025 0.005 0.01; do go $t $o 0 600 noise --task-args noise=$n; done
    for f in 0.2 0.3; do go $t $o 0 600 sparse --task-args subsample_fraction=$f; done
  done; done
fi

## capacity: Adam with larger networks and with warm restarts, in the time of GN on the paper's network, 600 s
if want capacity; then
  for t in rockerarm bunny; do
    go $t adam 0 600 capacity --task-args width=64
    go $t adam 0 600 capacity --task-args width=128
    go $t adam 0 600 capacity --task-args width=256 depth=4
    go $t adam 0 600 capacity --tag _restarts --restart-period 1000
  done
fi

## adam_long: Adam at 3600 s (three times the main budget, six for the INS), evaluations every 10 s
if want adam_long; then
  for t in catenoid enneper cone rockerarm bunny jeb; do go $t adam 0 3600 adam_3600 --eval-every 10; done
fi

## scaling: cost of one update against p and N (no training; results are merged into results/scaling/scaling.json)
if want scaling; then
  py -m experiments.scaling --variants dense woodbury adam residual
  py -m experiments.scaling --variants dense adam residual --ns 32000 64000 128000 --widths 16 32 64
  py -m experiments.scaling --variants cg
  py -m experiments.scaling --kinds ins --variants cg --ns 32000 64000 128000
fi

## figures: paper_figures/ from the finished runs (no training; the renders use the GPU and headless Chrome)
if want figures; then
  for t in catenoid enneper cone rockerarm bunny jeb; do py -m experiments.figures --task $t; done
  py -m experiments.figures --task jeb --suffix _strain_clip1000 --loss-terms strain
  py -m experiments.figures --task catenoid --runs gn_seed0 gn_seed4 --name nonunique --paper
  C=_pretrain_extent2.0_seed
  py -m experiments.figures --task catenoid --exp init_sweep --name init_sweep --starts --paper --labels "seed 0" "seed 1" "" "" "" \
    --runs gn_init_radius0.5${C}0 gn_init_radius0.5${C}1 gn_init_radius1.0${C}0 gn_init_radius1.543${C}0 gn_init_radius1.8${C}0
  py -m experiments.figures --task enneper --exp init_sweep --name init_sweep --starts --paper \
    --runs gn_seed0 gn_offset0.25_seed0 gn_offset0.5_seed0 gn_tilt15.0_seed0
  py -m experiments.seeds
  py -m experiments.scaling --plot-only --paper
fi
