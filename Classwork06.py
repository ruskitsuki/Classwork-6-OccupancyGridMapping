from robomaster import robot
import time
import csv
import math
from collections import deque
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("[INFO] matplotlib not installed.")

# ================= CONFIG =================
CONN_TYPE = "ap"
OUTPUT_DIR = Path(__file__).resolve().parent / "Classwork 6 OccupancyGridMapping"

GRID_SIZE = 4
MOVE_DISTANCE_M = 0.60

# --- SPEED CONFIG ---
MOVE_SPEED = 0.20            # ความเร็วเดินหน้า (m/s)
TURN_SPEED = 60              # ความเร็วหมุนตัว (deg/s)
TURN_SETTLE_TIME = 0.30      # เวลาพักหลังหมุนตัว (s)
SENSOR_SETTLE_TIME = 0.30    # เวลาพักอ่านเซนเซอร์ (s)
AUTO_START_DELAY = 1.0

DRIFT_COMP_Y = -0.015        # ชดเชยอาการเดินเอียงขวา (ติดลบจะดึงหุ่นไปทางซ้ายตอนเดิน)

# --- SENSOR CONFIG ---
FRONT_WALL_LIMIT_MM = 400.0  # ระยะ ToF ต่ำกว่า 40 cm (400 mm) ถือว่ามีกำแพงหน้า
FRONT_COLLISION_LIMIT_MM = 150.0  # ต่ำกว่า 15 cm ให้หยุดและถอยทันที
FRONT_SAFE_TARGET_MM = 175.0      # ระยะเป้าหมายหลังถอย 17.5 cm
FRONT_SAFE_MAX_MM = 200.0         # ช่วงปลอดภัยสูงสุด 20 cm
FRONT_RETREAT_SPEED_MPS = 0.08    # ถอยช้าเพื่อลดการกระชาก
FRONT_RETREAT_MIN_SPEED_MPS = 0.04
FRONT_RETREAT_RAMP_TIME_S = 0.25
FRONT_RETREAT_TIMEOUT_S = 3.0
FRONT_STOP_SETTLE_TIME_S = 0.30   # รอแรงเฉื่อยหยุดก่อนเริ่มถอย
FRONT_SAFE_CONFIRM_CYCLES = 3
FRONT_WRONG_DIRECTION_DROP_MM = 15.0
FRONT_WRONG_DIRECTION_CYCLES = 3
TOF_SUB_FREQ_HZ = 50              # ตรวจด้านหน้าแบบ real-time ทุกประมาณ 20 ms
TOF_CACHE_MAX_AGE_S = 0.15

IR_PORT = 1
IR_LEFT_ID = 2
IR_RIGHT_ID = 3
IR_ACTIVE_LOW = True          # การตั้งค่าเดิมที่ทดสอบแล้ว: IO=0 คือ WALL (Active Low)
IR_SAMPLE_COUNT = 5
IR_SAMPLE_DELAY = 0.008
IR_PROBE_DISTANCE_M = 0.05    # สไลด์ 5 cm ไปด้านที่ยังไม่พบกำแพงเพื่อตรวจซ้ำ
IR_PROBE_MAX_ATTEMPTS = 5     # ตรวจซ้ำสูงสุดด้านละ 5 ครั้ง (ระยะรวมสูงสุด 25 cm)
IR_PROBE_SPEED_MPS = 0.10     # drive_speed ไม่ถูกบังคับขั้นต่ำ 0.5 m/s เหมือน move()
IR_PROBE_MIN_SPEED_MPS = 0.04
IR_PROBE_RAMP_DISTANCE_M = 0.015  # ramp ช่วงต้นและท้าย 1.5 cm ลดการไถล
IR_PROBE_CONTROL_INTERVAL_S = 0.02
IR_PROBE_SETTLE_TIME = 0.15

# --- REAL-TIME SIDE CORRECTION ---
IR_SUB_FREQ_HZ = 50           # รับค่า Digital IR ทุกประมาณ 20 ms
IR_CACHE_MAX_AGE_S = 0.15
CONTROL_INTERVAL_S = 0.02     # อัปเดตคำสั่ง chassis ที่ 50 Hz
SIDE_ESCAPE_SPEED_MPS = 0.06  # ลดความเร็วสไลด์บนพื้นลื่น
SIDE_WALL_CONFIRM_CYCLES = 3  # ต้องพบ WALL ต่อเนื่องก่อนเริ่มหลบ
SIDE_CLEAR_CONFIRM_CYCLES = 3 # ต้องพบ CLEAR ต่อเนื่องก่อนเลิกหลบ
SIDE_SPEED_RAMP_MPS_PER_S = 0.30  # จำกัดอัตราการเปลี่ยน y_speed

# --- IMU HEADING HOLD ---
ATTITUDE_SUB_FREQ_HZ = 50
ATTITUDE_CACHE_MAX_AGE_S = 0.15
YAW_KP = 1.20                # z_speed ต่อ yaw error 1 degree
YAW_DEADBAND_DEG = 0.80
YAW_MAX_CORRECTION_DPS = 12.0
YAW_CORRECTION_SIGN = 1.0    # เปลี่ยนเป็น -1.0 หากหุ่นแก้ yaw ผิดทิศ
MOVE_COMMAND_TO_YAW_SIGN = -1.0  # RoboMaster: move(z=+90) ทำให้ attitude yaw ลดลง
YAW_ALIGN_TIMEOUT_S = 1.50
YAW_ALIGN_STABLE_CYCLES = 3

P_CELL_FREE = 0.05
P_IR_WALL = 0.90
P_IR_FREE = 0.10
FREE_THRESHOLD = 0.35
WALL_THRESHOLD = 0.65

START_CELL = (0, 0)
GOAL_CELL = (3, 0)
MAX_STEPS = 100

NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

DIR_NAME = {NORTH: "NORTH", EAST: "EAST", SOUTH: "SOUTH", WEST: "WEST"}
DIR_SHORT = {NORTH: "N", EAST: "E", SOUTH: "S", WEST: "W"}
HEADING_SYMBOL = {NORTH: "^", EAST: ">", SOUTH: "v", WEST: "<"}
DIRECTION_VECTOR = {NORTH: (0, 1), EAST: (1, 0), SOUTH: (0, -1), WEST: (-1, 0)}
OPPOSITE_DIRECTION = {NORTH: SOUTH, EAST: WEST, SOUTH: NORTH, WEST: EAST}

# Global Variables
robot_x = 0
robot_y = 0
heading = NORTH
tof_mm = 9999
tof_cache_timestamp = 0.0
tof_samples = deque(maxlen=5)
ir_left_value = None
ir_right_value = None
ir_cache_timestamp = 0.0
current_yaw = None
attitude_cache_timestamp = 0.0
heading_target_yaw = None

cell_log_odds = [[0.0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
wall_log_odds = [[[0.0, 0.0, 0.0, 0.0] for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

visited_cells = set()
scanned_cells = set()
movement_log = []
scan_log = []
map_history = []

fig = None
ax = None


# ================= MAP LOGIC =================
def probability_to_log_odds(probability):
    probability = max(0.001, min(0.999, probability))
    return math.log(probability / (1.0 - probability))

def log_odds_to_probability(log_odds):
    return 1.0 / (1.0 + math.exp(-log_odds))

def clamp_log_odds(value):
    return max(-4.0, min(4.0, value))

def inside_grid(x, y):
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

def neighbor_position(x, y, direction):
    dx, dy = DIRECTION_VECTOR[direction]
    return x + dx, y + dy

def get_cell_probability(x, y):
    return log_odds_to_probability(cell_log_odds[y][x])

def mark_cell_free(x, y):
    if inside_grid(x, y):
        cell_log_odds[y][x] = clamp_log_odds(
            cell_log_odds[y][x] + probability_to_log_odds(P_CELL_FREE)
        )

def get_wall_probability(x, y, direction):
    return log_odds_to_probability(wall_log_odds[y][x][direction])

def set_wall_log_odds_synced(x, y, direction, value):
    if not inside_grid(x, y):
        return
    value = clamp_log_odds(value)
    wall_log_odds[y][x][direction] = value
    nx, ny = neighbor_position(x, y, direction)
    if inside_grid(nx, ny):
        wall_log_odds[ny][nx][OPPOSITE_DIRECTION[direction]] = value

def update_wall_probability(x, y, direction, probability):
    if not inside_grid(x, y):
        return
    nx, ny = neighbor_position(x, y, direction)
    if not inside_grid(nx, ny):
        set_wall_log_odds_synced(x, y, direction, 4.0)
        return
        
    # 1. ถ้าเคยมาร์คว่าเป็นกำแพงชัวร์ๆ แล้ว ให้ล็อคค่าไว้เลย ไม่ต้องลดลง
    if wall_state(x, y, direction) == "WALL":
        set_wall_log_odds_synced(x, y, direction, 4.0)
        return

    new_value = wall_log_odds[y][x][direction] + probability_to_log_odds(probability)
    
    if new_value >= probability_to_log_odds(WALL_THRESHOLD):
        set_wall_log_odds_synced(x, y, direction, 4.0)
    else:
        set_wall_log_odds_synced(x, y, direction, new_value)

def initialize_outer_boundary():
    for i in range(GRID_SIZE):
        set_wall_log_odds_synced(i, 0, SOUTH, 4.0)
        set_wall_log_odds_synced(i, GRID_SIZE - 1, NORTH, 4.0)
        set_wall_log_odds_synced(0, i, WEST, 4.0)
        set_wall_log_odds_synced(GRID_SIZE - 1, i, EAST, 4.0)

def wall_state(x, y, direction):
    probability = get_wall_probability(x, y, direction)
    if probability >= WALL_THRESHOLD:
        return "WALL"
    if probability <= FREE_THRESHOLD:
        return "FREE"
    return "UNKNOWN"

def cell_state(x, y):
    probability = get_cell_probability(x, y)
    if probability <= FREE_THRESHOLD:
        return "Free"
    if probability >= WALL_THRESHOLD:
        return "Occupied"
    return "Unknown"


# ================= SENSOR & THRESHOLDS =================
def tof_callback(distance_info):
    global tof_mm, tof_cache_timestamp
    try:
        value = distance_info[0]
        if 30 <= value < 10000:
            tof_mm = value
            tof_samples.append(float(value))
            tof_cache_timestamp = time.monotonic()
    except Exception:
        pass


def adapter_callback(adapter_info):
    """เก็บ Digital IR ล่าสุดจาก Sensor Adapter แบบ real-time"""
    global ir_left_value, ir_right_value, ir_cache_timestamp
    try:
        io_data, _ = adapter_info
        left_index = (IR_LEFT_ID - 1) * 2 + (IR_PORT - 1)
        right_index = (IR_RIGHT_ID - 1) * 2 + (IR_PORT - 1)
        left_value = int(io_data[left_index])
        right_value = int(io_data[right_index])

        if left_value in (0, 1) and right_value in (0, 1):
            ir_left_value = left_value
            ir_right_value = right_value
            ir_cache_timestamp = time.monotonic()
    except (TypeError, ValueError, IndexError):
        pass


def attitude_callback(attitude_info):
    """เก็บค่า yaw ล่าสุดจาก IMU สำหรับควบคุมหัวให้ตรง"""
    global current_yaw, attitude_cache_timestamp
    try:
        yaw, _, _ = attitude_info
        current_yaw = float(yaw)
        attitude_cache_timestamp = time.monotonic()
    except (TypeError, ValueError):
        pass

def read_ir_once(sensor_adaptor, sensor_id, prefer_cache=True):
    """อ่าน Digital IR โดยเลือกใช้ real-time cache หรือ get_io() โดยตรง"""
    cache_age = time.monotonic() - ir_cache_timestamp
    if prefer_cache and cache_age <= IR_CACHE_MAX_AGE_S:
        if sensor_id == IR_LEFT_ID:
            return ir_left_value
        if sensor_id == IR_RIGHT_ID:
            return ir_right_value

    try:
        value = sensor_adaptor.get_io(id=sensor_id, port=IR_PORT)
        value = int(value) if value is not None else None
        return value if value in (0, 1) else None
    except Exception:
        return None

def ir_detects_wall(ir_value):
    if ir_value not in (0, 1):
        return None
    wall_value = 0 if IR_ACTIVE_LOW else 1
    return ir_value == wall_value

def read_ir_filtered(sensor_adaptor, sensor_id, prefer_cache=True):
    values = []
    for _ in range(IR_SAMPLE_COUNT):
        value = read_ir_once(sensor_adaptor, sensor_id, prefer_cache=prefer_cache)
        if value in (0, 1):
            values.append(value)
        time.sleep(IR_SAMPLE_DELAY)
    if not values:
        return None
    return 0 if values.count(0) > values.count(1) else 1

def read_tof_safe():
    tof_is_fresh = time.monotonic() - tof_cache_timestamp <= TOF_CACHE_MAX_AGE_S
    if not tof_is_fresh or not tof_samples:
        return 9999
    values = sorted(tof_samples)
    return values[len(values) // 2]

def update_wall_from_ir(x, y, direction, ir_value):
    has_wall = ir_detects_wall(ir_value)
    if has_wall is True:
        update_wall_probability(x, y, direction, P_IR_WALL)
    elif has_wall is False:
        update_wall_probability(x, y, direction, P_IR_FREE)


# ================= SIDE IR PROBE =================
def slide_slow(chassis, distance_y):
    """สไลด์ระยะสั้นด้วย speed ramp เพื่อลดการกระชากและการไถลของล้อ"""
    target_distance = abs(distance_y)
    if target_distance <= 0.0:
        return

    direction = 1.0 if distance_y > 0 else -1.0
    estimated_distance = 0.0
    previous_time = time.monotonic()

    try:
        while estimated_distance < target_distance:
            remaining_distance = target_distance - estimated_distance
            ramp_up = min(1.0, estimated_distance / IR_PROBE_RAMP_DISTANCE_M)
            ramp_down = min(1.0, remaining_distance / IR_PROBE_RAMP_DISTANCE_M)
            speed_scale = max(
                IR_PROBE_MIN_SPEED_MPS / IR_PROBE_SPEED_MPS,
                min(ramp_up, ramp_down)
            )
            speed_y = direction * IR_PROBE_SPEED_MPS * speed_scale
            command_z, _ = get_yaw_correction(heading_target_yaw)

            chassis.drive_speed(
                x=0,
                y=speed_y,
                z=command_z,
                timeout=max(0.10, IR_PROBE_CONTROL_INTERVAL_S * 3)
            )

            sleep_time = min(
                IR_PROBE_CONTROL_INTERVAL_S,
                remaining_distance / max(abs(speed_y), 0.001)
            )
            time.sleep(sleep_time)

            current_time = time.monotonic()
            estimated_distance += abs(speed_y) * (current_time - previous_time)
            previous_time = current_time
    finally:
        chassis.drive_speed(x=0, y=0, z=0)
        time.sleep(IR_PROBE_SETTLE_TIME)


def probe_side_wall(chassis, sensor_adaptor, sensor_id, side_name):
    """สไลด์ตรวจซ้ำจนพบกำแพงหรือครบจำนวนครั้ง แล้วกลับจุดเริ่มต้น"""
    # Probe ใช้ get_io() โดยตรง เพื่อไม่ให้ค่าจาก real-time cache ค้าง
    initial_value = read_ir_filtered(sensor_adaptor, sensor_id, prefer_cache=False)
    initial_wall = ir_detects_wall(initial_value)

    if initial_wall is True:
        print(f"[IR SCAN] {side_name} WALL detected (IO={initial_value})")
        return initial_value
    if initial_wall is None:
        print(f"[IR WARNING] {side_name} read failed; keep map UNKNOWN")
        return None

    step_y = IR_PROBE_DISTANCE_M if side_name == "RIGHT" else -IR_PROBE_DISTANCE_M
    total_y = 0.0
    final_value = initial_value
    print(
        f"[IR PROBE] {side_name} initially FREE; "
        f"check up to {IR_PROBE_MAX_ATTEMPTS} times"
    )

    try:
        for attempt in range(1, IR_PROBE_MAX_ATTEMPTS + 1):
            slide_slow(chassis, step_y)
            total_y += step_y

            probe_value = read_ir_filtered(
                sensor_adaptor,
                sensor_id,
                prefer_cache=False
            )
            if probe_value is None:
                print(f"[IR WARNING] {side_name} attempt {attempt} read failed")
                continue

            final_value = probe_value
            status = "WALL" if ir_detects_wall(probe_value) else "FREE"
            print(
                f"[IR PROBE] {side_name} attempt {attempt}/"
                f"{IR_PROBE_MAX_ATTEMPTS}: {status} (IO={probe_value})"
            )

            # พบกำแพงแล้ว ไม่ต้องสไลด์ตรวจต่อ
            if ir_detects_wall(probe_value):
                break
    except Exception as error:
        print(f"[IR PROBE ERROR] {side_name}: {error}")
    finally:
        if abs(total_y) > 0.0:
            try:
                completed_steps = int(round(abs(total_y / step_y)))
                # กลับด้วย speed ramp แบบเดียวกับขาออกเพื่อลดตำแหน่งคลาดเคลื่อน
                for _ in range(completed_steps):
                    slide_slow(chassis, -step_y)
            except Exception as error:
                print(f"[IR RETURN ERROR] {side_name}: {error}")

    return final_value


# ================= MOVEMENT =================
def turn_left(chassis):
    global heading, heading_target_yaw
    print(f"[TURN] LEFT from {DIR_NAME[heading]}")
    chassis.drive_speed(x=0, y=0, z=0)
    time.sleep(0.10)
    chassis.move(x=0, y=0, z=90, z_speed=TURN_SPEED).wait_for_completed()
    heading = (heading - 1) % 4
    if heading_target_yaw is not None:
        heading_target_yaw = normalize_angle_deg(
            heading_target_yaw + 90.0 * MOVE_COMMAND_TO_YAW_SIGN
        )
    time.sleep(TURN_SETTLE_TIME)
    align_heading_stationary(chassis)

def turn_right(chassis):
    global heading, heading_target_yaw
    print(f"[TURN] RIGHT from {DIR_NAME[heading]}")
    chassis.drive_speed(x=0, y=0, z=0)
    time.sleep(0.10)
    chassis.move(x=0, y=0, z=-90, z_speed=TURN_SPEED).wait_for_completed()
    heading = (heading + 1) % 4
    if heading_target_yaw is not None:
        heading_target_yaw = normalize_angle_deg(
            heading_target_yaw - 90.0 * MOVE_COMMAND_TO_YAW_SIGN
        )
    time.sleep(TURN_SETTLE_TIME)
    align_heading_stationary(chassis)

def turn_back(chassis):
    print(f"[TURN] BACK from {DIR_NAME[heading]} using two RIGHT turns")
    # คำสั่ง 180° ครั้งเดียวอาจไม่เริ่มหลัง speed control จึงแบ่งเป็น 90° สองครั้ง
    turn_right(chassis)
    turn_right(chassis)

def turn_to_heading(chassis, target_heading):
    difference = (target_heading - heading) % 4
    print(
        f"[TURN PLAN] Current={DIR_NAME[heading]} -> "
        f"Target={DIR_NAME[target_heading]} | difference={difference}"
    )
    if difference == 1:
        turn_right(chassis)
    elif difference == 2:
        turn_back(chassis)
    elif difference == 3:
        turn_left(chassis)
    else:
        print("[TURN] Already facing target heading; no turn needed")


def get_realtime_side_speed():
    """คำนวณความเร็วแกน y จากค่า IR ล่าสุด โดยไม่หยุดการเดินหน้า"""
    cache_age = time.monotonic() - ir_cache_timestamp
    if cache_age > IR_CACHE_MAX_AGE_S:
        return None, "IR_STALE"

    left_wall = ir_detects_wall(ir_left_value)
    right_wall = ir_detects_wall(ir_right_value)

    if left_wall is True and right_wall is True:
        return 0.0, "BOTH_WALLS"
    if right_wall is True:
        # ตามแกนเดิมของโปรเจกต์: +y = ขวา ดังนั้นหนีกำแพงขวาด้วย -y
        return -SIDE_ESCAPE_SPEED_MPS, "AVOID_RIGHT"
    if left_wall is True:
        return SIDE_ESCAPE_SPEED_MPS, "AVOID_LEFT"
    return None, "CLEAR"


def retreat_front_to_safe_distance(chassis):
    """ถอยด้วย drive_speed จนถึง 17.5 cm โดยไม่แก้พิกัดกริด"""
    start_time = time.monotonic()
    start_distance_mm = read_tof_safe()
    wrong_direction_cycles = 0
    safe_cycles = 0

    if start_distance_mm == 9999:
        print("[FRONT RETREAT ERROR] No fresh ToF data; robot remains stopped")
        return False

    print(f"[FRONT RETREAT] Start at {start_distance_mm:.1f} mm")

    try:
        while time.monotonic() - start_time < FRONT_RETREAT_TIMEOUT_S:
            front_mm = read_tof_safe()
            if front_mm == 9999:
                print("[FRONT RETREAT ERROR] ToF data lost; emergency stop")
                return False

            if front_mm >= FRONT_SAFE_TARGET_MM:
                chassis.drive_speed(x=0, y=0, z=0)
                safe_cycles += 1
                if safe_cycles >= FRONT_SAFE_CONFIRM_CYCLES:
                    status = "SAFE" if front_mm <= FRONT_SAFE_MAX_MM else "SAFE BUT FAR"
                    print(f"[FRONT RETREAT] {status}: ToF={front_mm:.1f} mm")
                    return True
                time.sleep(CONTROL_INTERVAL_S)
                continue

            safe_cycles = 0

            # หากระยะลดลงต่อเนื่อง แสดงว่าหุ่นกำลังเคลื่อนผิดทิศหรือยังไหลไปข้างหน้า
            if front_mm < start_distance_mm - FRONT_WRONG_DIRECTION_DROP_MM:
                wrong_direction_cycles += 1
            else:
                wrong_direction_cycles = 0

            if wrong_direction_cycles >= FRONT_WRONG_DIRECTION_CYCLES:
                print(
                    f"[FRONT RETREAT ERROR] ToF decreased to {front_mm:.1f} mm; "
                    "wrong direction detected, emergency stop"
                )
                return False

            elapsed = time.monotonic() - start_time
            ramp_scale = min(1.0, elapsed / FRONT_RETREAT_RAMP_TIME_S)
            retreat_speed = max(
                FRONT_RETREAT_MIN_SPEED_MPS,
                FRONT_RETREAT_SPEED_MPS * ramp_scale
            )
            chassis.drive_speed(
                x=-retreat_speed,
                y=0,
                z=0,
                timeout=max(0.10, CONTROL_INTERVAL_S * 3)
            )
            time.sleep(CONTROL_INTERVAL_S)

        print("[FRONT RETREAT ERROR] Timeout; emergency stop")
        return False
    finally:
        chassis.drive_speed(x=0, y=0, z=0)


def normalize_angle_deg(angle):
    """จำกัดมุมให้อยู่ในช่วง -180 ถึง 180 degree"""
    return (angle + 180.0) % 360.0 - 180.0


def get_yaw_correction(target_yaw):
    """คำนวณ z_speed เพื่อรักษา yaw เริ่มต้นระหว่างเดินตรง"""
    if target_yaw is None or current_yaw is None:
        return 0.0, None

    cache_age = time.monotonic() - attitude_cache_timestamp
    if cache_age > ATTITUDE_CACHE_MAX_AGE_S:
        return 0.0, None

    yaw_error = normalize_angle_deg(target_yaw - current_yaw)
    if abs(yaw_error) <= YAW_DEADBAND_DEG:
        return 0.0, yaw_error

    correction = YAW_CORRECTION_SIGN * YAW_KP * yaw_error
    correction = max(
        -YAW_MAX_CORRECTION_DPS,
        min(YAW_MAX_CORRECTION_DPS, correction)
    )
    return correction, yaw_error


def align_heading_stationary(chassis):
    """จัดหัวให้ตรงกับ yaw เป้าหมายขณะหยุด ก่อนเริ่มการเคลื่อนที่ถัดไป"""
    if heading_target_yaw is None:
        return False

    deadline = time.monotonic() + YAW_ALIGN_TIMEOUT_S
    stable_cycles = 0

    try:
        while time.monotonic() < deadline:
            command_z, yaw_error = get_yaw_correction(heading_target_yaw)
            if yaw_error is None:
                print("[HEADING ALIGN WARNING] IMU yaw unavailable")
                return False

            if abs(yaw_error) <= YAW_DEADBAND_DEG:
                stable_cycles += 1
                if stable_cycles >= YAW_ALIGN_STABLE_CYCLES:
                    print(
                        f"[HEADING ALIGN] yaw={current_yaw:.2f}, "
                        f"error={yaw_error:.2f} deg"
                    )
                    return True
            else:
                stable_cycles = 0

            chassis.drive_speed(
                x=0,
                y=0,
                z=command_z,
                timeout=max(0.10, CONTROL_INTERVAL_S * 3)
            )
            time.sleep(CONTROL_INTERVAL_S)
    finally:
        chassis.drive_speed(x=0, y=0, z=0)

    print("[HEADING ALIGN WARNING] Alignment timeout")
    return False


def move_forward_realtime(chassis):
    """เดินหน้า พร้อมแก้ตำแหน่งด้านข้างและ yaw แบบ real-time"""
    global heading_target_yaw

    move_duration = MOVE_DISTANCE_M / MOVE_SPEED
    drift_speed_y = DRIFT_COMP_Y / move_duration
    deadline = time.monotonic() + move_duration
    previous_state = None
    confirmed_side_state = "CLEAR"
    candidate_side_state = None
    candidate_cycles = 0
    command_y = drift_speed_y
    yaw_is_fresh = (
        current_yaw is not None
        and time.monotonic() - attitude_cache_timestamp <= ATTITUDE_CACHE_MAX_AGE_S
    )
    if heading_target_yaw is None and yaw_is_fresh:
        heading_target_yaw = current_yaw
    target_yaw = heading_target_yaw

    if target_yaw is None:
        print("[HEADING HOLD WARNING] No fresh IMU yaw; z correction disabled")
    else:
        print(f"[HEADING HOLD] Target yaw={target_yaw:.2f} deg")

    try:
        while True:
            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0.0:
                break

            # หยุดก่อนชน แล้วถอยเป็น physical correction โดยไม่เปลี่ยน robot_x/robot_y
            tof_is_fresh = time.monotonic() - tof_cache_timestamp <= TOF_CACHE_MAX_AGE_S
            if tof_is_fresh and 30 <= tof_mm < FRONT_COLLISION_LIMIT_MM:
                chassis.drive_speed(x=0, y=0, z=0)
                print(f"[FRONT SAFETY] ToF={tof_mm:.1f} mm < {FRONT_COLLISION_LIMIT_MM:.1f} mm")
                time.sleep(FRONT_STOP_SETTLE_TIME_S)
                retreat_front_to_safe_distance(chassis)
                break

            _, raw_side_state = get_realtime_side_speed()

            # Debounce + hysteresis: เปลี่ยนสถานะเมื่อค่าคงที่ต่อเนื่องเท่านั้น
            if raw_side_state == confirmed_side_state:
                candidate_side_state = None
                candidate_cycles = 0
            else:
                if raw_side_state == candidate_side_state:
                    candidate_cycles += 1
                else:
                    candidate_side_state = raw_side_state
                    candidate_cycles = 1

                confirm_cycles = (
                    SIDE_WALL_CONFIRM_CYCLES
                    if raw_side_state in ("AVOID_LEFT", "AVOID_RIGHT", "BOTH_WALLS")
                    else SIDE_CLEAR_CONFIRM_CYCLES
                )
                if candidate_cycles >= confirm_cycles:
                    confirmed_side_state = raw_side_state
                    candidate_side_state = None
                    candidate_cycles = 0

            if confirmed_side_state == "AVOID_RIGHT":
                target_speed_y = -SIDE_ESCAPE_SPEED_MPS
            elif confirmed_side_state == "AVOID_LEFT":
                target_speed_y = SIDE_ESCAPE_SPEED_MPS
            elif confirmed_side_state == "BOTH_WALLS":
                target_speed_y = 0.0
            else:
                target_speed_y = drift_speed_y

            # Ramp y_speed เพื่อไม่ให้ล้อ Mecanum กระชากบนพื้นลื่น
            max_speed_change = SIDE_SPEED_RAMP_MPS_PER_S * CONTROL_INTERVAL_S
            speed_error = target_speed_y - command_y
            command_y += max(
                -max_speed_change,
                min(max_speed_change, speed_error)
            )
            command_z, yaw_error = get_yaw_correction(target_yaw)

            if confirmed_side_state != previous_state:
                yaw_text = "N/A" if yaw_error is None else f"{yaw_error:.2f} deg"
                print(
                    f"[REAL-TIME IR] {confirmed_side_state} | "
                    f"L={ir_left_value}, R={ir_right_value}, "
                    f"y_speed={command_y:.2f} m/s, yaw_error={yaw_text}"
                )
                previous_state = confirmed_side_state

            chassis.drive_speed(
                x=MOVE_SPEED,
                y=command_y,
                z=command_z,
                timeout=max(0.10, CONTROL_INTERVAL_S * 3)
            )
            time.sleep(min(CONTROL_INTERVAL_S, remaining_time))
    finally:
        chassis.drive_speed(x=0, y=0, z=0)

def direction_between(current, target):
    dx = target[0] - current[0]
    dy = target[1] - current[1]
    if (dx, dy) == (0, 1): return NORTH
    if (dx, dy) == (1, 0): return EAST
    if (dx, dy) == (0, -1): return SOUTH
    if (dx, dy) == (-1, 0): return WEST
    raise ValueError("Path cells are not adjacent")


# ================= SCAN =================
def scan_current_cell_smart(chassis, sensor_adaptor, step):
    cell = (robot_x, robot_y)

    print("\n=================================================")
    print(f"SMART SCAN CELL {cell} | Heading: {DIR_NAME[heading]}")
    print("=================================================")

    front_dir = heading
    left_dir = (heading - 1) % 4
    right_dir = (heading + 1) % 4
    back_dir = (heading + 2) % 4

    left_state_before = wall_state(robot_x, robot_y, left_dir)
    right_state_before = wall_state(robot_x, robot_y, right_dir)
    front_state_before = wall_state(robot_x, robot_y, front_dir)

    # ตรวจเฉพาะ UNKNOWN; ด้านที่รู้แล้วว่า FREE/WALL จะไม่ Probe ซ้ำ
    if right_state_before == "UNKNOWN":
        right_ir = probe_side_wall(chassis, sensor_adaptor, IR_RIGHT_ID, "RIGHT")
        update_wall_from_ir(robot_x, robot_y, right_dir, right_ir)
    else:
        right_ir = None
        print(f"[IR SCAN] RIGHT already {right_state_before}; skip")

    if left_state_before == "UNKNOWN":
        left_ir = probe_side_wall(chassis, sensor_adaptor, IR_LEFT_ID, "LEFT")
        update_wall_from_ir(robot_x, robot_y, left_dir, left_ir)
    else:
        left_ir = None
        print(f"[IR SCAN] LEFT already {left_state_before}; skip")

    if right_state_before == "UNKNOWN" or left_state_before == "UNKNOWN":
        align_heading_stationary(chassis)

    if front_state_before == "UNKNOWN":
        front_mm = read_tof_safe()
        if front_mm < FRONT_WALL_LIMIT_MM:
            update_wall_probability(robot_x, robot_y, front_dir, P_IR_WALL)
            print(f"[ToF SCAN] Front WALL detected ({front_mm:.1f} mm < {FRONT_WALL_LIMIT_MM} mm)")
        else:
            update_wall_probability(robot_x, robot_y, front_dir, P_IR_FREE)
            print(f"[ToF SCAN] Front FREE ({front_mm:.1f} mm >= {FRONT_WALL_LIMIT_MM} mm)")
        front_log = round(front_mm, 1)
    else:
        front_mm = None
        front_log = f"SKIP_{front_state_before}"
        print(f"[ToF SCAN] Front already {front_state_before}; skip")

    # ช่องด้านหลังคือทางที่หุ่นเพิ่งเดินผ่านมา จึงเป็น FREE โดยไม่ต้องอ่าน Sensor
    if wall_state(robot_x, robot_y, back_dir) == "UNKNOWN":
        update_wall_probability(robot_x, robot_y, back_dir, P_IR_FREE)

    mark_cell_free(robot_x, robot_y)
    scanned_cells.add(cell)

    scan_log.append({
        "Step": step,
        "Robot Pos(x,y)": f"({robot_x},{robot_y})",
        "IR Left": left_ir if left_state_before == "UNKNOWN" else f"SKIP_{left_state_before}",
        "ToF": front_log,
        "IR Right": right_ir if right_state_before == "UNKNOWN" else f"SKIP_{right_state_before}",
        "Map Status": cell_state(robot_x, robot_y)
    })

    save_map_snapshot(step, f"SMART_SCAN_{robot_x}_{robot_y}")
    display_realtime()


# ================= AUTO NAVIGATION =================
def get_next_cell_autonomous():
    directions = [
        ((heading + 1) % 4, "RIGHT"),
        (heading, "FRONT"),
        ((heading - 1) % 4, "LEFT")
    ]

    for direction, label in directions:
        nx, ny = neighbor_position(robot_x, robot_y, direction)
        if inside_grid(nx, ny) and (nx, ny) not in visited_cells and (nx, ny) != GOAL_CELL:
            if wall_state(robot_x, robot_y, direction) == "FREE":
                print(f"[AUTO] {label} clear: {DIR_NAME[direction]}")
                return nx, ny

    for direction, label in directions:
        nx, ny = neighbor_position(robot_x, robot_y, direction)
        if inside_grid(nx, ny) and (nx, ny) not in visited_cells and (nx, ny) != GOAL_CELL:
            if wall_state(robot_x, robot_y, direction) == "UNKNOWN":
                print(f"[AUTO] Explore {label}: {DIR_NAME[direction]}")
                return nx, ny

    queue = deque([(robot_x, robot_y)])
    parent = {(robot_x, robot_y): None}
    target = None

    while queue:
        curr_x, curr_y = queue.popleft()
        if (curr_x, curr_y) not in visited_cells and (curr_x, curr_y) != GOAL_CELL:
            target = (curr_x, curr_y)
            break
        
        for d in (NORTH, EAST, SOUTH, WEST):
            if wall_state(curr_x, curr_y, d) in ("FREE", "UNKNOWN"):
                nx, ny = neighbor_position(curr_x, curr_y, d)
                if inside_grid(nx, ny) and (nx, ny) not in parent:
                    parent[(nx, ny)] = (curr_x, curr_y)
                    queue.append((nx, ny))
    
    if target:
        curr = target
        while parent[curr] != (robot_x, robot_y):
            curr = parent[curr]
        print(f"[AUTO] Backtracking to unvisited cell {target}, moving to {curr}")
        return curr

    if (robot_x, robot_y) != GOAL_CELL:
        print("[AUTO] All reachable cells visited! Heading to GOAL_CELL...")
        queue = deque([(robot_x, robot_y)])
        parent = {(robot_x, robot_y): None}

        while queue:
            curr_x, curr_y = queue.popleft()
            if (curr_x, curr_y) == GOAL_CELL:
                break
            
            for d in (NORTH, EAST, SOUTH, WEST):
                # หลังสำรวจครบ ให้กลับ Goal ผ่านเส้นทางที่ยืนยันว่า FREE แล้วเท่านั้น
                if wall_state(curr_x, curr_y, d) == "FREE":
                    nx, ny = neighbor_position(curr_x, curr_y, d)
                    if inside_grid(nx, ny) and (nx, ny) not in parent:
                        parent[(nx, ny)] = (curr_x, curr_y)
                        queue.append((nx, ny))
        
        if GOAL_CELL in parent:
            curr = GOAL_CELL
            while parent[curr] != (robot_x, robot_y):
                curr = parent[curr]
            print(f"[AUTO] Moving towards GOAL_CELL {GOAL_CELL}, next step {curr}")
            return curr
        else:
            print("[AUTO] GOAL_CELL is unreachable!")
            return None

    print("[AUTO] All reachable cells visited and currently at GOAL_CELL!")
    return None


def move_to_cell(chassis, sensor_adaptor, target_cell, step):
    global robot_x, robot_y

    current_cell = (robot_x, robot_y)
    original_heading = heading
    target_heading = direction_between(current_cell, target_cell)

    turn_to_heading(chassis, target_heading)
    time.sleep(SENSOR_SETTLE_TIME)

    target_wall_state = wall_state(robot_x, robot_y, target_heading)
    if target_wall_state == "WALL":
        print(f"[MOVE BLOCKED] {DIR_NAME[target_heading]} is already WALL")
        turn_to_heading(chassis, original_heading)
        return False

    if target_wall_state == "UNKNOWN":
        front_mm = read_tof_safe()
        if front_mm < FRONT_WALL_LIMIT_MM:
            print(f"[WALL DETECTED] {DIR_NAME[target_heading]} blocked by ToF ({front_mm:.1f} mm < {FRONT_WALL_LIMIT_MM} mm). Recalculating path...")
            set_wall_log_odds_synced(robot_x, robot_y, target_heading, 4.0)
            # พบกำแพงหลังหมุน: กลับไปยัง heading เดิมก่อนคำนวณเส้นทางใหม่
            turn_to_heading(chassis, original_heading)
            time.sleep(SENSOR_SETTLE_TIME)
            return False

        update_wall_probability(robot_x, robot_y, target_heading, P_IR_FREE)
        print(f"[MOVE CHECK] {DIR_NAME[target_heading]} confirmed FREE by ToF")
    else:
        print(f"[MOVE CHECK] {DIR_NAME[target_heading]} already FREE; skip scan")

    move_forward_realtime(chassis)

    robot_x, robot_y = target_cell
    visited_cells.add(target_cell)

    movement_log.append({
        "Step": step,
        "From X": current_cell[0],
        "From Y": current_cell[1],
        "To X": robot_x,
        "To Y": robot_y
    })

    save_map_snapshot(step, f"MOVE_TO_{robot_x}_{robot_y}")
    time.sleep(SENSOR_SETTLE_TIME)
    display_realtime()

    return True


# ================= GUI =================
def init_gui():
    global fig, ax
    if not GUI_AVAILABLE:
        return
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.show(block=False)

def draw_gui():
    if not GUI_AVAILABLE or fig is None:
        return

    ax.clear()
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            ax.text(x + 0.5, y + 0.18, f"({x},{y})", ha="center", va="center", fontsize=8)

            if (x, y) == START_CELL:
                ax.text(x + 0.5, y + 0.5, "S", ha="center", va="center", fontsize=18)
            elif (x, y) == GOAL_CELL:
                ax.text(x + 0.5, y + 0.5, "G", ha="center", va="center", fontsize=18)

            for direction in (NORTH, EAST, SOUTH, WEST):
                state = wall_state(x, y, direction)
                if state == "FREE":
                    continue
                linewidth = 5 if state == "WALL" else 1
                linestyle = "-" if state == "WALL" else "--"

                if direction == NORTH:
                    ax.plot([x, x + 1], [y + 1, y + 1], lw=linewidth, ls=linestyle)
                elif direction == EAST:
                    ax.plot([x + 1, x + 1], [y, y + 1], lw=linewidth, ls=linestyle)
                elif direction == SOUTH:
                    ax.plot([x, x + 1], [y, y], lw=linewidth, ls=linestyle)
                elif direction == WEST:
                    ax.plot([x, x], [y, y + 1], lw=linewidth, ls=linestyle)

    ax.text(robot_x + 0.5, robot_y + 0.55, HEADING_SYMBOL[heading], ha="center", va="center", fontsize=30)
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect("equal")

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.01)

def display_realtime():
    draw_gui()


# ================= SAVE =================
def save_map_snapshot(step, event):
    row = {
        "Step": step,
        "Event": event,
        "Robot X": robot_x,
        "Robot Y": robot_y,
        "Heading": DIR_NAME[heading]
    }
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            row[f"Cell({x},{y})"] = round(get_cell_probability(x, y), 3)
            for direction in (NORTH, EAST, SOUTH, WEST):
                row[f"Cell({x},{y})_{DIR_SHORT[direction]}"] = round(
                    get_wall_probability(x, y, direction), 3
                )
    map_history.append(row)

def save_csv(filename, rows):
    if not rows:
        return
    try:
        output_path = OUTPUT_DIR / filename
        with open(output_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    except Exception as e:
        print(f"[WARNING] Failed to save {filename}: {e}")

def save_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_csv("movement_log.csv", movement_log)
    save_csv("experiment_results.csv", scan_log)
    save_csv("map_history.csv", map_history)

    if GUI_AVAILABLE and fig is not None:
        try:
            draw_gui()
            fig.savefig(OUTPUT_DIR / "final_wall_map.png", dpi=200, bbox_inches="tight")
            print("[INFO] Saved final_wall_map.png successfully.")
        except (Exception, KeyboardInterrupt) as e:
            print(f"[WARNING] Could not save final_wall_map.png: {e}")


# ================= MAIN =================
def main():
    global robot_x, robot_y, heading, heading_target_yaw

    initialize_outer_boundary()
    visited_cells.add(START_CELL)
    mark_cell_free(robot_x, robot_y)

    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type=CONN_TYPE)

    chassis = ep_robot.chassis
    distance_sensor = ep_robot.sensor
    sensor_adaptor = ep_robot.sensor_adaptor

    try:
        distance_sensor.sub_distance(freq=TOF_SUB_FREQ_HZ, callback=tof_callback)
        sensor_adaptor.sub_adapter(freq=IR_SUB_FREQ_HZ, callback=adapter_callback)
        chassis.sub_attitude(freq=ATTITUDE_SUB_FREQ_HZ, callback=attitude_callback)
        init_gui()
        display_realtime()

        time.sleep(AUTO_START_DELAY)

        if (
            current_yaw is not None
            and time.monotonic() - attitude_cache_timestamp <= ATTITUDE_CACHE_MAX_AGE_S
        ):
            heading_target_yaw = current_yaw
            print(f"[HEADING REFERENCE] NORTH yaw={heading_target_yaw:.2f} deg")
        else:
            print("[HEADING REFERENCE WARNING] IMU yaw not ready")

        step = 0
        scan_current_cell_smart(chassis, sensor_adaptor, step)

        while True:
            step += 1
            if step > MAX_STEPS:
                print("[WARNING] Maximum steps reached.")
                break

            target_cell = get_next_cell_autonomous()
            if target_cell is None:
                print("[INFO] Exploration and Exit complete, or no reachable cells.")
                break

            if not move_to_cell(chassis, sensor_adaptor, target_cell, step):
                print("[INFO] Obstacle found! Recalculating route...")
                continue

            scan_current_cell_smart(chassis, sensor_adaptor, step)

        if (robot_x, robot_y) == GOAL_CELL:
            print("[SUCCESS] GOAL REACHED")
        else:
            print("[INFO] MAPPING STOPPED")

    except KeyboardInterrupt:
        print("[INFO] EMERGENCY STOP")
    except Exception as error:
        print(f"[ERROR] {error}")
    finally:
        try:
            chassis.drive_speed(x=0, y=0, z=0)
        except Exception:
            pass

        try:
            distance_sensor.unsub_distance()
        except Exception:
            pass

        try:
            sensor_adaptor.unsub_adapter()
        except Exception:
            pass

        try:
            chassis.unsub_attitude()
        except Exception:
            pass

        try:
            save_all()
        except Exception:
            pass

        ep_robot.close()


if __name__ == "__main__":
    main()
