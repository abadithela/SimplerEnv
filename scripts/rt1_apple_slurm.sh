# shader_dir=rt means that we turn on ray-tracing rendering; this is quite crucial for the open / close drawer task as policies often rely on shadows to infer depth

ckpt_path=$1   # Model
env_name=$2 # Experiment environment
urdf_version=$3 # URDF version

EXTRA_ARGS="--enable-raytracing --additional-env-build-kwargs station_name=mk_station_recolor light_mode=simple disable_bad_material=True urdf_version=${urdf_version} model_ids=baked_apple_v2"

  # A0
python simpler_env/main_inference.py --policy-model rt1 --ckpt-path ${ckpt_path} \
--robot google_robot_static \
--control-freq 3 --sim-freq 513 --max-episode-steps 200 \
--env-name ${env_name} --scene-name dummy_drawer \
--robot-init-x 0.645 0.705 5 --robot-init-y -0.179 -0.085 5 \
--robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.03 -0.03 1 \
--obj-init-x-range -0.08 -0.02 5 --obj-init-y-range -0.02 0.08 5 \
--rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_a0.png \
${EXTRA_ARGS} &
# B0
python simpler_env/main_inference.py --policy-model rt1 --ckpt-path ${ckpt_path} \
--robot google_robot_static \
--control-freq 3 --sim-freq 513 --max-episode-steps 200 \
--env-name ${env_name} --scene-name dummy_drawer \
--robot-init-x 0.644 0.705 5 --robot-init-y -0.086 0.117 5 \
--robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
--obj-init-x-range -0.08 -0.02 5 --obj-init-y-range -0.02 0.08 5 \
--rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_b0.png \
${EXTRA_ARGS} &
# C0
python simpler_env/main_inference.py --policy-model rt1 --ckpt-path ${ckpt_path} \
--robot google_robot_static \
--control-freq 3 --sim-freq 513 --max-episode-steps 200 \
--env-name ${env_name} --scene-name dummy_drawer \
--robot-init-x 0.644 0.705 5 --robot-init-y 0.118 0.224 5 \
--robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
--obj-init-x-range -0.08 -0.02 5 --obj-init-y-range -0.02 0.08 5 \
--rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/open_drawer_c0.png \
${EXTRA_ARGS}







