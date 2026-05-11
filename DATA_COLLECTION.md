# Data Collection Setup

Commands assume repo root.

## 1. Common Repo Setup

1. Create and activate the Python environment.

   ```bash
   mamba create -n anycar python=3.10
   mamba activate anycar
   ```

2. Install repo packages.

   ```bash
   pip install -r requirements.txt
   ```

3. Export the repo path.

   ```bash
   export CAR_PATH=$(pwd)
   ```
 ation/data/<timestamp>-numeric_sim/
   ```

4. Optional: `--data-dir /path/to/output`

## 3. MuJoCo Collection

1. Install MuJoCo.

   The repo does not pin a MuJoCo version. It uses DeepMind `mujoco`, not `mujoco-py`.

   ```bash
   pip install mujoco
   ```

2. Test import.

   ```bash
   python -c "import mujoco; print(mujoco.__version__)"
   ```

3. Run the MuJoCo collector.

   ```bash
   python car_collect/mujoco_collect/parallel_mujoco_collect_data.py --simend 20000 --episodes 1
   ```

4. Output:

   ```text
   car_foundation/car_foundation/data/mujoco_sim_debugging/
   ```

5. Optional: `--data-dir /path/to/output --no-render --debug-plots`

## 4. Isaac Sim Collection

1. Install Isaac Sim 2023.1.1 from the direct Linux archive.

   ```bash
   sudo apt update
   sudo apt install -y wget unzip

   mkdir -p ~/Downloads ~/.local/share/ov/pkg/isaac-sim-2023.1.1

   wget -O ~/Downloads/isaac-sim-standalone-2023.1.1-linux-x86_64.zip \
     "https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone%402023.1.1-rc.8%2B2023.1.688.573e0291.tc.linux-x86_64.release.zip"

   unzip ~/Downloads/isaac-sim-standalone-2023.1.1-linux-x86_64.zip \
     -d ~/.local/share/ov/pkg/isaac-sim-2023.1.1

   cd ~/.local/share/ov/pkg/isaac-sim-2023.1.1
   chmod +x omni.isaac.sim.post.install.sh
   ./omni.isaac.sim.post.install.sh
   ```

2. Configure the `anycar_is` conda environment to use Isaac Sim with plain `python3`.

   These commands assume Isaac Sim is installed at `~/.local/share/ov/pkg/isaac-sim-2023.1.1`.

   ```bash
   conda create -n anycar_is python=3.10 -y
   conda activate anycar_is

   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt

   ISAAC_SIM_PATH="$HOME/.local/share/ov/pkg/isaac-sim-2023.1.1"

   conda env config vars set -n anycar_is \
     CARB_APP_PATH="$ISAAC_SIM_PATH/kit" \
     ISAAC_PATH="$ISAAC_SIM_PATH" \
     EXP_PATH="$ISAAC_SIM_PATH/apps" \
     PYTHONPATH="$ISAAC_SIM_PATH/exts/omni.isaac.kit:$ISAAC_SIM_PATH/exts/omni.isaac.gym:$ISAAC_SIM_PATH/kit/kernel/py:$ISAAC_SIM_PATH/kit/plugins/bindings-python:$ISAAC_SIM_PATH/exts/omni.isaac.lula/pip_prebundle:$ISAAC_SIM_PATH/exts/omni.exporter.urdf/pip_prebundle:$ISAAC_SIM_PATH/kit/exts/omni.kit.pip_archive/pip_prebundle:$ISAAC_SIM_PATH/exts/omni.isaac.core_archive/pip_prebundle:$ISAAC_SIM_PATH/exts/omni.isaac.ml_archive/pip_prebundle:$ISAAC_SIM_PATH/exts/omni.pip.compute/pip_prebundle:$ISAAC_SIM_PATH/exts/omni.pip.cloud/pip_prebundle:$ISAAC_SIM_PATH/extscache/omni.pip.torch-2_0_1-2.0.2+105.1.lx64/torch-2-0-1:$ISAAC_SIM_PATH/kit/python/lib/python3.10/site-packages" \
     LD_LIBRARY_PATH="$ISAAC_SIM_PATH:$ISAAC_SIM_PATH/exts/omni.usd.schema.isaac/plugins/IsaacSensorSchema/lib:$ISAAC_SIM_PATH/exts/omni.usd.schema.isaac/plugins/RangeSensorSchema/lib:$ISAAC_SIM_PATH/exts/omni.isaac.lula/pip_prebundle:$ISAAC_SIM_PATH/exts/omni.exporter.urdf/pip_prebundle:$ISAAC_SIM_PATH/kit:$ISAAC_SIM_PATH/kit/kernel/plugins:$ISAAC_SIM_PATH/kit/libs/iray:$ISAAC_SIM_PATH/kit/plugins:$ISAAC_SIM_PATH/kit/plugins/bindings-python:$ISAAC_SIM_PATH/kit/plugins/carb_gfx:$ISAAC_SIM_PATH/kit/plugins/rtx:$ISAAC_SIM_PATH/kit/plugins/gpu.foundation"
   ```

   Reactivate the env so conda applies the configured variables:

   ```bash
   conda deactivate
   conda activate anycar_is
   ```

3. Test Isaac Sim with conda `python3`.

   ```bash
   python3 -c "from omni.isaac.kit import SimulationApp; app = SimulationApp({'headless': True}); app.close()"
   ```

4. Run the Isaac Sim collector.

   ```bash
   cd /home/rwik/research/jumpracing/anycar
   python3 car_collect/isaacsim_collect/isaacsim_collect_data.py --episodes 1
   ```

   For headless data collection:

   ```bash
   python3 car_collect/isaacsim_collect/isaacsim_collect_data.py --headless --episodes 1
   ```

5. Output:

   ```text
   car_foundation/car_foundation/data/isaac_sim_trash/
   ```

6. Optional: `--data-dir /path/to/output --no-render --debug-plots --camera-offset -3 -3 3`

## 5. Isaac Lab Parallel Collection

This uses the newer Isaac Lab install in `anycar_lab` and runs multiple F1Tenth cars in one Isaac Lab process.

```bash
conda activate anycar_lab
cd /home/rwik/research/jumpracing/anycar

python3 car_collect/isaaclab_collect/parallel_isaaclab_collect_data.py \
  --headless \
  --num-envs 64 \
  --episodes 10 \
  --simend 2000
```

Output defaults to:

```text
car_foundation/car_foundation/data/isaaclab_sim_debugging/
```

Each environment writes one raw float32 `.bin` per episode. Rows are 15D:

```text
pos(3), quat_wxyz(4), vel_world(3), angvel_world(3), throttle, steer
```

Optional useful args:

```bash
--data-dir /path/to/output
--num-envs 128
--device cuda:0
--usd-name F1Tenth_lecar.usd
--seed 3
--dt 0.02
--physics-dt 0.01
--env-spacing 0
--max-throttle 1.0
--ground-static-friction 1.5
--ground-dynamic-friction 1.5
--wheel-static-friction 1.5
--wheel-dynamic-friction 1.5
--speed-filter-alpha 0.25
--action-filter-alpha 0.25
--lower-vel 0.7
--upper-vel 2.0
--min-track-scale 1
--max-track-scale 5
--min-kp 6.0
--max-kp 10.0
--min-kd 0.5
--max-kd 1.5
```

## 6. Assetto Corsa Linux Collection

This uses a separate env:

```bash
conda create -n anycar_ac python=3.10 -y
conda activate anycar_ac
python -m pip install --upgrade "pip==23.3.2" "setuptools==65.5.0" "wheel==0.38.4"
python -m pip install -r car_dynamics/car_dynamics/envs/assetto_corsa/requirements.txt
python -m pip install evdev
python -m pip install -e car_dataset -e car_dynamics -e car_collect -e car_planner -e car_foundation
python -m pip install -e car_dynamics/car_dynamics/envs/assetto_corsa/assetto_corsa_gym
```

Install Steam:

```bash
sudo dpkg --add-architecture i386
sudo add-apt-repository multiverse
sudo apt update
sudo apt install -y steam-installer
steam
```

In Steam:

```text
Login
Install Assetto Corsa
Steam > Settings > Compatibility > enable Steam Play for all titles
Launch Assetto Corsa once, then close it
```

Install GE-Proton:

```bash
mkdir -p ~/.steam/debian-installation/compatibilitytools.d
mkdir -p ~/Downloads
PROTONUP_URL=$(python - <<'PY'
import json, urllib.request
release = json.load(urllib.request.urlopen("https://api.github.com/repos/DavidoTek/ProtonUp-Qt/releases/latest"))
for asset in release["assets"]:
    if asset["name"].endswith(".AppImage") and "x86_64" in asset["name"]:
        print(asset["browser_download_url"])
        break
PY
)
wget -O ~/Downloads/ProtonUp-Qt.AppImage "$PROTONUP_URL"
chmod +x ~/Downloads/ProtonUp-Qt.AppImage
~/Downloads/ProtonUp-Qt.AppImage
```

In ProtonUp-Qt:

```text
Install for: Steam
Add version
Compatibility tool: GE-Proton
GE-Proton9-2
Install
```

In Steam:

```text
Restart Steam
Assetto Corsa > Properties > Compatibility > Force GE-Proton9-2
```

If Assetto Corsa was first launched with another Proton version:

```bash
rm -rf ~/.steam/debian-installation/steamapps/compatdata/244210
```

Launch Assetto Corsa once with `GE-Proton9-2`, then close it.

Install fonts:

```bash
AC_PREFIX="$HOME/.steam/debian-installation/steamapps/compatdata/244210/pfx"
mkdir -p "$AC_PREFIX/drive_c/windows/Fonts"

wget -O "$AC_PREFIX/drive_c/windows/Fonts/Verdana.ttf" \
  https://raw.githubusercontent.com/matomo-org/travis-scripts/master/fonts/Verdana.ttf
wget -O "$AC_PREFIX/drive_c/windows/Fonts/segoeuiz.ttf" \
  "https://raw.githubusercontent.com/xamarin/evolve-presentation-template/master/Fonts/Segoe%20UI/segoeuiz.ttf"
wget -O "$AC_PREFIX/drive_c/windows/Fonts/segoeui.ttf" \
  "https://raw.githubusercontent.com/xamarin/evolve-presentation-template/master/Fonts/Segoe%20UI/segoeui.ttf"
wget -O "$AC_PREFIX/drive_c/windows/Fonts/verdanai.ttf" \
  https://raw.githubusercontent.com/dolbydu/font/master/Sans/Verdana/verdanai.ttf
```

Get upstream Linux plugin/config files:

```bash
mkdir -p third_party
git clone https://github.com/dasGringuen/assetto_corsa_gym.git third_party/assetto_corsa_gym
```

Copy plugin and controller files:

```bash
AC_ROOT="$HOME/.steam/debian-installation/steamapps/common/assettocorsa"

mkdir -p "$AC_ROOT/apps/python"
cp -r third_party/assetto_corsa_gym/assetto_corsa_gym/AssettoCorsaPlugin/plugins/sensors_par \
  "$AC_ROOT/apps/python/"

mkdir -p "$AC_ROOT/cfg/controllers/savedsetups"
mkdir -p "$AC_ROOT/system/x64"

cp third_party/assetto_corsa_gym/assetto_corsa_gym/AssettoCorsaPlugin/windows-libs/Vjoy_linux.ini \
  "$AC_ROOT/cfg/controllers/savedsetups/"
cp third_party/assetto_corsa_gym/assetto_corsa_gym/AssettoCorsaPlugin/windows-libs/WASD.ini \
  "$AC_ROOT/cfg/controllers/savedsetups/"
cp -r third_party/assetto_corsa_gym/assetto_corsa_gym/AssettoCorsaPlugin/windows-libs/DLLs \
  "$AC_ROOT/system/x64/"
cp -r third_party/assetto_corsa_gym/assetto_corsa_gym/AssettoCorsaPlugin/windows-libs/Lib \
  "$AC_ROOT/system/x64/"
```

Install virtual controller:

```bash
sudo apt-get install -y xboxdrv
sudo xboxdrv --daemon --silent --mimic-xpad --type xbox360 --dbus disabled
```

Assetto Corsa settings:

```text
Enable Python app: sensors_par
Load controller setup: Vjoy Linux
Set framerate limit: 50 FPS
Mode: Challenge > Hotlap
Automatic Gearbox: ON
Traction Control: OFF
Stability Control: OFF
ABS: OFF 
Mechanical Damage: OFF
Tyre Wear: OFF
Fuel Consumption: OFF
```

Run collection after the hotlap session is loaded:

```bash
conda activate anycar_ac
python car_collect/assetto_corsa_collect/acc_pure_pursuit.py
```

The collector starts `xboxdrv` if needed and sends gear-up automatically. If the car is stuck in neutral:

```bash
python car_collect/assetto_corsa_collect/ac_gear_up.py
```

Note: this repo does not ship the `*_0.1m.pkl` track occupancy grids. The env falls back to Assetto Corsa's own off-track state when those files are missing.

## 6. Output Format

Each collector writes `.pkl` files containing a `CarDataset` object. The dataset includes:

- `car_params`
- `steer`
- `throttle`
- position
- orientation quaternion
- linear velocity
- linear acceleration
- angular velocity
- reference trajectory fields
- `lap_end`

## 7. Equal Simulator Split

Use equal timesteps:

```python
total_timesteps = simend * episodes
```

Small test:

```python
simend = 2000
episodes = 5
```

Paper-scale `100M` total split across numeric, MuJoCo, Isaac:

```text
100,000,000 / 3 = 33,333,333 timesteps per simulator
```

Use this for each collector:

```bash
--simend 2000 --episodes 16667
```

This gives:

```text
33,334,000 timesteps per simulator
100,002,000 timesteps total
```

## 8. Changes Made In This Repo

Numeric / MuJoCo / Isaac Sim:

```text
car_collect/numeric_collect/collect_data_gym.py
- added --simend, --episodes, --data-dir
- made Ray optional/lazy so one episode runs without Ray workers

car_collect/mujoco_collect/parallel_mujoco_collect_data.py
- added --simend, --episodes, --data-dir, --no-render, --debug-plots
- made Ray optional/lazy for non-render parallel collection

car_collect/isaacsim_collect/isaacsim_collect_data.py
- added --simend, --episodes, --data-dir, --no-render, --debug-plots
- disabled cone spawning for clean collection

car_dynamics/car_dynamics/models_jax/__init__.py
- made DynamicsJax lazy so DBM collection does not import Transformer Engine unless needed
```

Isaac Sim visual changes:

```text
car_dynamics/car_dynamics/envs/isaac_sim/car_isaac.py
- restored stable camera-follow mode
- added scene lighting
- made ground plane black
- removed cones from collection path
```

Assetto Corsa collection changes:

```text
car_collect/assetto_corsa_collect/acc_pure_pursuit.py
- removed Ray dependency
- added --simend, --episodes, --data-dir
- added controller args: --lower-vel, --upper-vel, --lookahead, --kp, --kd, --steer-sign, --steer-gain
- unwraps Gym TimeLimit so old Gym 4-return API works
- starts xboxdrv automatically
- sends gear-up automatically
- saves data even if the episode terminates out-of-track

car_collect/assetto_corsa_collect/ac_gear_up.py
- helper script to send Xbox A / gear-up through evdev

car_dynamics/car_dynamics/envs/assetto_corsa/config.yml
- set track to imola
- set car to ks_alfa_33_stradale
- disabled send_reset_at_start

car_dynamics/car_dynamics/envs/assetto_corsa/assetto_corsa_gym/AssettoCorsaConfigs/tracks/config.yaml
- added imola track metadata

car_dynamics/car_dynamics/envs/assetto_corsa/assetto_corsa_gym/AssettoCorsaConfigs/tracks/imola.csv
car_dynamics/car_dynamics/envs/assetto_corsa/assetto_corsa_gym/AssettoCorsaConfigs/tracks/imola-racing_line.csv
- added imola centerline and racing line files

car_dynamics/car_dynamics/envs/assetto_corsa/assetto_corsa_gym/AssettoCorsaEnv/ac_env.py
- missing *_0.1m.pkl track grid no longer crashes
- falls back to Assetto Corsa's numberOfTyresOut off-track state

car_dynamics/car_dynamics/envs/assetto_corsa/assetto_corsa_gym/AssettoCorsaEnv/car_control.py
- imports vjoy_linux on Linux instead of Windows vJoy DLL

car_dynamics/car_dynamics/envs/assetto_corsa/assetto_corsa_gym/AssettoCorsaEnv/vjoy_linux.py
- Linux evdev backend for steering, throttle, brake, and gear-up

car_dynamics/car_dynamics/envs/assetto_corsa/assetto_corsa_gym/assetto-corsa-autonomous-racing-plugin/plugins/sensors_par/sensors_par.py
- keeps ego_server.tick() running after static info is collected
- guards ext_isAltPressed for Linux/Proton compatibility

car_dynamics/car_dynamics/envs/assetto_corsa/assetto_corsa_gym/assetto-corsa-autonomous-racing-plugin/plugins/sensors_par/vjoy_linux.py
- copy-pasteable Linux vJoy backend for the installed sensors_par plugin
```

Copy the repo vJoy backend into the installed Assetto Corsa plugin:

```bash
AC_ROOT="$HOME/.steam/debian-installation/steamapps/common/assettocorsa"
cp car_dynamics/car_dynamics/envs/assetto_corsa/assetto_corsa_gym/assetto-corsa-autonomous-racing-plugin/plugins/sensors_par/vjoy_linux.py \
  "$AC_ROOT/apps/python/sensors_par/vjoy_linux.py"
```

Patch the installed `sensors_par.py` by copying the repo plugin file:

```bash
AC_ROOT="$HOME/.steam/debian-installation/steamapps/common/assettocorsa"
cp car_dynamics/car_dynamics/envs/assetto_corsa/assetto_corsa_gym/assetto-corsa-autonomous-racing-plugin/plugins/sensors_par/sensors_par.py \
  "$AC_ROOT/apps/python/sensors_par/sensors_par.py"
```
