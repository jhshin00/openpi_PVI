import datetime
import glob
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import threading
from queue import Queue, Empty

import numpy as np
import tyro
import h5py
import pygame


from gello.agents.agent import BimanualAgent, DummyAgent
from gello.agents.gello_agent import GelloAgent
from gello.data_utils.format_obs import save_frame
from gello.env import RobotEnv
from gello.robots.robot import PrintRobot
from gello.zmq_core.robot_node import ZMQClientRobot
from gello.zmq_core.camera_node import ZMQClientCamera, ZMQServerCamera
from gello.cameras.camera import CameraDriver  # Protocol

from ur3_forward_kinematics import calculate_cartesian_action
from forward_kinematics import forward_kinematics_ur3, calculate_cartesian_action_rotvec, step_joint_from_policy_action

import pyrealsense2 as rs
import cv2

def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)

reset_q_global = np.deg2rad([0, -90, -90, -90, 90, 90])

def go_to_reset(env, reset_q, steps=120, sleep_dt=0.01, per_step_limit=0.05):
    """env.get_obs()에서 현재 q를 읽어 reset_q까지 선형 보간 이동"""
    obs = env.get_obs()
    q_now = obs["joint_positions"].copy()
    # 길이 맞추기 (그리퍼 포함 7D면 그대로, 6D면 6D까지만)
    n = min(len(q_now), len(reset_q))
    q_now = q_now[:n]
    q_goal = np.array(reset_q[:n], dtype=float)

    for t in np.linspace(0.0, 1.0, steps):
        q_cmd = (1 - t) * q_now + t * q_goal
        # per-step joint limit (너무 큰 점프 방지)
        obs = env.get_obs()
        q_cur = obs["joint_positions"][:n]
        dq = q_cmd - q_cur
        m = np.max(np.abs(dq))
        if m > per_step_limit:
            dq = dq / m * per_step_limit
            q_cmd = q_cur + dq
        env.step(q_cmd if len(obs["joint_positions"])==n else np.concatenate([q_cmd, obs["joint_positions"][n:]]))
        time.sleep(sleep_dt)

def go_to_reset_fast(env, reset_q, dt=0.01, vmax=1.2, amax=6.0, tol=1e-3, max_time=8.0):
    """
    속도/가속도 제한을 적용한 시간기반 리셋.
    - dt: 제어 주기 (초)
    - vmax: 관절 속도 상한 [rad/s]
    - amax: 관절 가속도 상한 [rad/s^2]
    - tol: 종료 기준 (최대 관절 오차)
    - max_time: 안전 종료 시간
    """
    obs = env.get_obs()
    q_cur = obs["joint_positions"].copy()
    n = min(len(q_cur), len(reset_q))
    q_goal = np.array(reset_q[:n], dtype=float)
    q_cur = q_cur[:n]

    v = np.zeros(n)  # 현재 관절 속도 상태(명령 속도)
    t0 = time.time()

    while True:
        # 종료 조건
        err = q_goal - q_cur
        if np.max(np.abs(err)) < tol:
            break
        if time.time() - t0 > max_time:
            print(f"[go_to_reset_fast] timeout after {time.time()-t0:.2f}s (max|e|={np.max(np.abs(err)):.4f})")
            break

        # 가속도 제한으로 속도 업데이트 (목표부호로 가속)
        # v <- v + clip( amax*dt in error 방향 )
        desired_sign = np.sign(err)
        v = v + desired_sign * (amax * dt)

        # 속도 상한 제한
        v = np.clip(v, -vmax, vmax)

        # 감속(브레이크) 거리 고려: 남은 거리로부터 허용속도 제한 (v^2 <= 2*a*|e|)
        # → overshoot 방지(최소 제동거리 보장)
        v_brake = np.sqrt(2.0 * amax * np.abs(err))  # 각 관절별 허용 속도 상한
        v = np.clip(v, -v_brake, v_brake)

        # 위치 업데이트(명령 생성)
        q_cmd = q_cur + v * dt

        # env에 반영 (그리퍼 등 나머지 차원 유지)
        obs_full = env.get_obs()
        q_full = obs_full["joint_positions"].copy()
        q_full[:n] = q_cmd
        env.step(q_full)

        # 다음 루프 준비
        q_cur = env.get_obs()["joint_positions"][:n]
        # 주기 유지(오버런이면 sleep 생략)
        t_loop_start = time.time()
        t_elapsed = time.time() - t_loop_start
        if dt - t_elapsed > 0:
            time.sleep(dt - t_elapsed)


@dataclass
class Args:
    agent: str = "gello"
    robot_port: int = 6001
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    hostname: str = "127.0.0.1"
    robot_type: str = None  # only needed for quest agent or spacemouse agent
    hz: int = 30
    start_joints: Optional[Tuple[float, ...]] = None

    gello_port: Optional[str] = None
    mock: bool = False
    use_save_interface: bool = True
    data_dir: str = "/ssd1/data_pi0.5/pick_and_place"
    bimanual: bool = False
    verbose: bool = False
    cut_frames: bool = False # Only needed when you want to save a fixed number of frames 
    frames: int = 84


class RealSenseDriver(CameraDriver):
    def __init__(self, serial: str, width=640, height=480, fps=30):
        self.serial = serial
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(cfg)

    def read(self, img_size=None):
        frames = self.pipeline.wait_for_frames()
        color = frames.get_color_frame()
        img = np.asanyarray(color.get_data())
        if img_size is not None:
            img = cv2.resize(img, (img_size[1], img_size[0]))
        return img

    def __str__(self):
        return f"RealSense({self.serial})"

def start_server(port, driver):
    print(f"[CameraServer] start {driver}")
    server = ZMQServerCamera(driver, port=port)
    server.serve()

class AsyncCamera:
    def __init__(self, client: ZMQClientCamera, target_size=(224,224), crop_center=None, crop_box=None):
        self.client = client
        self.target_size = target_size
        self.frame = None
        self.lock = threading.Lock()

        if crop_box is not None:
            self.x1, self.x2, self.y1, self.y2 = crop_box
            self.use_crop = True
        elif crop_center is not None:
            h, w = 400, 400
            cy, cx = crop_center
            self.x1 = max(cx - w // 2, 0)
            self.x2 = min(cx + w // 2, 640)
            self.y1 = max(cy - h // 2, 0)
            self.y2 = min(cy + h // 2, 480)
            self.use_crop = True
        else:
            cy, cx = None, None
            self.x1 = self.x2 = self.y1 = self.y2 = None
            self.use_crop = False

        # 데몬 쓰레드로 백그라운드에서 프레임 갱신
        t = threading.Thread(target=self._update_loop, daemon=True)
        t.start()

    def _update_loop(self):
        while True:
            raw = self.client.read()
            if self.use_crop:                
                crop = raw[self.y1:self.y2, self.x1:self.x2]
                img = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = cv2.resize(raw, self.target_size, interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            with self.lock:
                self.frame = rgb

    def read(self):
        # Return lastest frame
        with self.lock:
            return self.frame

##요주의 suboptimal traj: 0107_151546 0107_152116
def main(args):
    if args.mock:
        robot_client = PrintRobot(8, dont_print=True)
        camera_clients = {}
    else:
        ctx = rs.context()
        devs = ctx.query_devices()
        if len(devs) < 2:
            print("두 대 이상의 RealSense가 연결되어 있지 않습니다.")
            return
        serials = [dev.get_info(rs.camera_info.serial_number) for dev in devs[:2]]
        drivers = [RealSenseDriver(s) for s in serials]
        ports = [5000, 5001]
        threads = []
        for port, drv in zip(ports, drivers):
            t = threading.Thread(target=start_server, args=(port, drv), daemon=True)
            t.start()
            threads.append(t)

        time.sleep(1)  # 바인딩 대기

        # Connect clients
        client1 = ZMQClientCamera(port=5000)
        client2 = ZMQClientCamera(port=5001)

        camera_clients = {
            # "wrist": ZMQClientCamera(port=args.wrist_camera_port, host=args.hostname),
            # "base": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),

            "base": AsyncCamera(client1, crop_box=(0, 500, 0, 480)),
            "wrist": AsyncCamera(client2),
        }
        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=camera_clients)

    if args.bimanual:
        if args.agent == "gello":
            # dynamixel control box port map (to distinguish left and right gello)
            right = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBG6A-if00-port0"
            left = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBEIA-if00-port0"
            left_agent = GelloAgent(port=left)
            right_agent = GelloAgent(port=right)
            agent = BimanualAgent(left_agent, right_agent)
        elif args.agent == "quest":
            from gello.agents.quest_agent import SingleArmQuestAgent

            left_agent = SingleArmQuestAgent(robot_type=args.robot_type, which_hand="l")
            right_agent = SingleArmQuestAgent(
                robot_type=args.robot_type, which_hand="r"
            )
            agent = BimanualAgent(left_agent, right_agent)

        elif args.agent == "spacemouse":
            from gello.agents.spacemouse_agent import SpacemouseAgent

            left_path = "/dev/hidraw0"
            right_path = "/dev/hidraw1"
            left_agent = SpacemouseAgent(
                robot_type=args.robot_type, device_path=left_path, verbose=args.verbose
            )
            right_agent = SpacemouseAgent(
                robot_type=args.robot_type,
                device_path=right_path,
                verbose=args.verbose,
                invert_button=True,
            )
            agent = BimanualAgent(left_agent, right_agent)
        else:
            raise ValueError(f"Invalid agent name for bimanual: {args.agent}")

        # System setup specific. This reset configuration works well on our setup. If you are mounting the robot
        # differently, you need a separate reset joint configuration.
        reset_joints_left = np.deg2rad([0, -90, -90, -90, 90, 0, 0])
        reset_joints_right = np.deg2rad([0, -90, 90, -90, -90, 0, 0])
        reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
        curr_joints = env.get_obs()["joint_positions"]
        max_delta = (np.abs(curr_joints - reset_joints)).max()
        steps = min(int(max_delta / 0.01), 100)

        for jnt in np.linspace(curr_joints, reset_joints, steps):
            env.step(jnt)
    else:
        if args.agent == "gello":
            gello_port = args.gello_port
            if gello_port is None:
                usb_ports = glob.glob("/dev/serial/by-id/*")
                print(f"Found {len(usb_ports)} ports")
                if len(usb_ports) > 0:
                    gello_port = usb_ports[0]
                    print(f"using port {gello_port}")
                else:
                    raise ValueError(
                        "No gello port found, please specify one or plug in gello"
                    )
            if args.start_joints is None:
                reset_joints = np.deg2rad(
                    # [0, -90, 90, -90, -90, 0, 0]
                    [0, -90, -90, -90, 90, 180]
                )  # Change this to your own reset joints
            else:
                reset_joints = args.start_joints
            agent = GelloAgent(port=gello_port, start_joints=args.start_joints)
            curr_joints = env.get_obs()["joint_positions"]
            if reset_joints.shape == curr_joints.shape:
                max_delta = (np.abs(curr_joints - reset_joints)).max()
                steps = min(int(max_delta / 0.01), 100)

                for jnt in np.linspace(curr_joints, reset_joints, steps):
                    env.step(jnt)
                    time.sleep(0.001)
        elif args.agent == "quest":
            from gello.agents.quest_agent import SingleArmQuestAgent

            agent = SingleArmQuestAgent(robot_type=args.robot_type, which_hand="l")
        elif args.agent == "spacemouse":
            from gello.agents.spacemouse_agent import SpacemouseAgent

            agent = SpacemouseAgent(robot_type=args.robot_type, verbose=args.verbose)
        elif args.agent == "dummy" or args.agent == "none":
            agent = DummyAgent(num_dofs=robot_client.num_dofs())
        elif args.agent == "policy":
            raise NotImplementedError("add your imitation policy here if there is one")
        else:
            raise ValueError("Invalid agent name")

    # going to start position
    print("Going to start position")
    start_pos = agent.act(env.get_obs())
    obs = env.get_obs()
    joints = obs["joint_positions"]

    abs_deltas = np.abs(start_pos - joints)
    id_max_joint_delta = np.argmax(abs_deltas)

    max_joint_delta = 0.8
    if abs_deltas[id_max_joint_delta] > max_joint_delta:
        id_mask = abs_deltas > max_joint_delta
        print()
        ids = np.arange(len(id_mask))[id_mask]
        for i, delta, joint, current_j in zip(
            ids,
            abs_deltas[id_mask],
            start_pos[id_mask],
            joints[id_mask],
        ):
            print(
                f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
            )
        return

    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    assert len(start_pos) == len(
        joints
    ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

    max_delta = 0.05
    for _ in range(25):
        obs = env.get_obs()
        command_joints = agent.act(obs)
        current_joints = obs["joint_positions"]
        delta = command_joints - current_joints
        max_joint_delta = np.abs(delta).max()
        if max_joint_delta > max_delta:
            delta = delta / max_joint_delta * max_delta
        env.step(current_joints + delta)

    obs = env.get_obs()
    joints = obs["joint_positions"]
    action = agent.act(obs)
    if (action - joints > 0.5).any():
        print("Action is too big")

        # print which joints are too big
        joint_index = np.where(action - joints > 0.8)
        for j in joint_index:
            print(
                f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}"
            )
        exit()

    if args.use_save_interface:
        from gello.data_utils.keyboard_interface import KBReset

        kb_interface = KBReset()

    print_color("\nStart 🚀🚀🚀", color="green", attrs=("bold",))

    action = obs['joint_positions']
    
    dt = 1.0 / args.hz
    Kp = 8.0            
    deadband = 0.003        # rad
    vmax = 1.5              # rad/s
    amax = 10.0             # rad/s^2
    a_speedj = 10.0
    step = 0

    qd_prev = np.zeros(6)
    save_path = None
    buffer: List[Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]] = []
    recording: bool = False

    t_next = time.perf_counter()
    while True:
        current_joints = obs["joint_positions"].copy()

        action = agent.act(obs)

        # FK        (Policy Input) -> current EE pose
        ee_pose = np.hstack((forward_kinematics_ur3(current_joints[:6]), current_joints[-1]))

        # Action    (Policy Output) -> Cartesian delta
        ee_action = calculate_cartesian_action_rotvec(current_joints[:6], action)
        target_joints = step_joint_from_policy_action(current_joints[:6], ee_action)
        target_joints = action.copy()
        # PD control
        err = target_joints[:6] - current_joints[:6]
        err[np.abs(err) < deadband] = 0.0

        # Instead of deviding by dt, we multiply by Kp (1/s)
        qd = Kp * err

        # clip velocity (필수)
        qd = np.clip(qd, -vmax, vmax)

        # clip accel (필수)
        dq = qd - qd_prev
        dq = np.clip(dq, -amax*dt, amax*dt)
        qd = qd_prev + dq
        qd_prev = qd

        obs = env.step_velocity(qd, a=a_speedj, t=dt, gripper=action[6])


        if args.use_save_interface:
            state = kb_interface.update()
            if state == "start":
                dt_time = datetime.datetime.now()
                save_path = (
                    Path(args.data_dir).expanduser()
                    / dt_time.strftime("%m%d_%H%M%S")
                )
                save_path.mkdir(parents=True, exist_ok=True)
                print(f"Recording to {save_path}")
                buffer.clear()
                recording = True

            elif state == "reset":
                print("Resetting")
                recording = False
                buffer.clear()

                reset_q_global = np.deg2rad([0, -90, -90, -90, 90, 90])
                reset_q_global = np.concatenate([reset_q_global, [0.0]])  # 7D (그리퍼 포함)
                q_cur = env.get_obs()["joint_positions"]

                max_time = 2.0
                t0 = time.time()
                alpha = 0.5
                dt = 0.01

                while np.max(np.abs(q_cur - reset_q_global)) > 1e-2 and (time.time() - t0) < max_time:
                    q_cmd = (1 - alpha) * q_cur + alpha * reset_q_global
                    env.step(q_cmd)              # 목표 관절 그대로 커맨드
                    obs = env.get_obs()
                    
                    time.sleep(dt)
                    q_cur = obs["joint_positions"]
                    
                print("Reset done")
                kb_interface.pressed_last = None
                state = "normal"

            elif state == "save":
                assert save_path is not None, "something went wrong"
                buffer.append((obs.copy(), action.copy(), ee_pose.copy(), ee_action.copy()))  

            elif state == "normal":                     # End of recording: flush buffer to HDF5
                if recording:
                    if save_path and buffer:
                        # # Create HDF5 file
                        h5_file = save_path / "data.hdf5"
                        with h5py.File(h5_file, 'w') as f:
                            grp = f.create_group('data')
                            # keys from first obs
                            keys = list(buffer[0][0].keys())
                            for key in keys:
                                # stack obs values
                                data_arr = np.stack([b[0][key] for b in buffer], axis=0)
                                grp.create_dataset(key, data=data_arr)

                            acts = np.stack([b[1] for b in buffer], axis=0)
                            ee_pose = np.stack([b[2] for b in buffer], axis=0)
                            ee_action = np.stack([b[3] for b in buffer], axis=0)
                            grp.create_dataset('joint_actions', data=acts)
                            grp.create_dataset('ee_pose', data=ee_pose)
                            grp.create_dataset('ee_delta_action', data=ee_action)

                        # 동일 폴더에 base 와 wrist 각 카메라별로 비디오 생성
                        import cv2
                        fps = 30
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                        for cam in ('base','wrist'):
                            #  프레임 모으기
                            frames = [b[0][f'{cam}_rgb'] for b in buffer]
                            h, w = frames[0].shape[:2]
                            video_path = save_path / f"{cam}_rgb.mp4"
                            vw = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
                            for f_rgb in frames:
                                f_bgr = cv2.cvtColor(f_rgb, cv2.COLOR_RGB2BGR)
                                vw.write(f_bgr)
                            vw.release()
                    # reset
                    recording = False
                    buffer.clear()
                    save_path = None
            else:
                raise ValueError(f"Invalid state {state}")

        t_next += dt
        t_now = time.perf_counter()
        sleep_time = t_next - t_now
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            t_next = t_now
        
        if step % args.hz == 0:
            print(f"[Timing] loop period ~ {dt*1000:.1f} ms target")
        step += 1


if __name__ == "__main__":
    main(tyro.cli(Args))
