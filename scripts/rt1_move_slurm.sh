
gpu_id="0"

env_name=MoveNearGoogleBakedTexInScene-vIROM
# env_name=MoveNearGoogleBakedTexInScene-v1
scene_name=google_pick_coke_can_1_v4
rgb_overlay_path=./ManiSkill2_real2sim/data/real_inpainting/google_move_near_real_eval_1.png

ckpt_path=$1   # Model
urdf_version=$2 # URDF version
shift 2

epi_int=("$@")
echo "Sequence: ${epi_int[@]}"

CUDA_VISIBLE_DEVICES=${gpu_id} 
python simpler_env/main_inference.py --policy-model rt1 --ckpt-path ${ckpt_path} \
--robot google_robot_static \
--control-freq 3 --sim-freq 513 --max-episode-steps 80 \
--env-name ${env_name} --scene-name ${scene_name} \
--rgb-overlay-path ${rgb_overlay_path} \
--robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range ${epi_int[0]} ${epi_int[1]} \
--robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
--additional-env-build-kwargs urdf_version=${urdf_version} \
--additional-env-save-tags baked_except_bpb_orange & 
python simpler_env/main_inference.py --policy-model rt1 --ckpt-path ${ckpt_path} \
--robot google_robot_static \
--control-freq 3 --sim-freq 513 --max-episode-steps 80 \
--env-name ${env_name} --scene-name ${scene_name} \
--rgb-overlay-path ${rgb_overlay_path} \
--robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range ${epi_int[1]} ${epi_int[2]} \
--robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
--additional-env-build-kwargs urdf_version=${urdf_version} \
--additional-env-save-tags baked_except_bpb_orange & 
python simpler_env/main_inference.py --policy-model rt1 --ckpt-path ${ckpt_path} \
--robot google_robot_static \
--control-freq 3 --sim-freq 513 --max-episode-steps 80 \
--env-name ${env_name} --scene-name ${scene_name} \
--rgb-overlay-path ${rgb_overlay_path} \
--robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range ${epi_int[2]} ${epi_int[3]} \
--robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
--additional-env-build-kwargs urdf_version=${urdf_version} \
--additional-env-save-tags baked_except_bpb_orange & 
python simpler_env/main_inference.py --policy-model rt1 --ckpt-path ${ckpt_path} \
--robot google_robot_static \
--control-freq 3 --sim-freq 513 --max-episode-steps 80 \
--env-name ${env_name} --scene-name ${scene_name} \
--rgb-overlay-path ${rgb_overlay_path} \
--robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range ${epi_int[3]} ${epi_int[4]} \
--robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
--additional-env-build-kwargs urdf_version=${urdf_version} \
--additional-env-save-tags baked_except_bpb_orange & 
python simpler_env/main_inference.py --policy-model rt1 --ckpt-path ${ckpt_path} \
--robot google_robot_static \
--control-freq 3 --sim-freq 513 --max-episode-steps 80 \
--env-name ${env_name} --scene-name ${scene_name} \
--rgb-overlay-path ${rgb_overlay_path} \
--robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range ${epi_int[4]} ${epi_int[5]} \
--robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
--additional-env-build-kwargs urdf_version=${urdf_version} \
--additional-env-save-tags baked_except_bpb_orange 
