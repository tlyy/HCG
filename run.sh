for game in Asterix; do
    for i in 1 2 3; do
        python -m bbf.train \
                --agent=BBF \
                --gin_files=bbf/configs/BBF.gin \
                --base_dir=exp/icassp/all_sp/all_cum_grama_u50_rand_cyc/$game/$i \
                --gin_bindings="DataEfficientAtariRunner.game_name = '$game'"
    done
done


# for game in Kangaroo; do
#     for i in 1 2 3 4 5; do
#         python -m bbf.train \
#                 --agent=BBF \
#                 --gin_files=bbf/configs/BBF.gin \
#                 --base_dir=exp/orth_loss0.00001/$game/$i \
#                 --gin_bindings="DataEfficientAtariRunner.game_name = '$game'"
#     done
# done

# Assault Asterix BankHeist BattleZone ChopperCommand CrazyClimber Freeway Frostbite Gopher Kangaroo KungFuMaster Pong Qbert UpNDown
# Boxing Breakout DemonAttack Jamesbond Krull RoadRunner
# Alien Amidar Hero MsPacman PrivateEye Seaquest

# export MUJOCO_GL=osmesa
# for i in 1 2 3; do
#     for game in cartpole-swingup reacher-easy cheetah-run finger-spin ball_in_cup-catch walker-walk; do
#         python -m continuous_control.train \
#                 --save_dir=exp_con/drq/$game/$i \
#                 --env_name $game \
#                 --max_steps 100000
#     done
# done