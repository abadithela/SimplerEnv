

gpu_id="0,1,2,3"

declare -a arr=("./checkpoints/rt_1_tf_trained_for_000400120/" \
                "./checkpoints/rt_1_tf_trained_for_000058240/" \
                "./checkpoints/rt_1_x_tf_trained_for_002272480_step/" \
                "./checkpoints/rt_1_tf_trained_for_000001120/" )

# lr_switch=laying horizontally but flipped left-right to match real eval; upright=standing; laid_vertically=laying vertically
declare -a coke_can_options_arr=("lr_switch=True" "upright=True" "laid_vertically=True")

# URDF variations
declare -a urdf_version_arr=(None "recolor_tabletop_visual_matching_1" "recolor_tabletop_visual_matching_2" "recolor_cabinet_visual_matching_1")

env_name=GraspSingleOpenedCokeCanInScene-v0
scene_name=google_pick_coke_can_1_v4
rgb_overlay_path=./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png

for ckpt_path in "${arr[@]}"; do echo "$ckpt_path"; done



for urdf_version in "${urdf_version_arr[@]}";

do for coke_can_option in "${coke_can_options_arr[@]}";

do for ckpt_path in "${arr[@]}";

# Setting a denser grid of obj x and y evals.
do CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model rt1 --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name ${env_name} --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.349 -0.121 $1 20 --obj-init-y -0.019 0.419 20 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --additional-env-build-kwargs ${coke_can_option} urdf_version=${urdf_version} ;

done

done

done
