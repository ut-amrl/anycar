import struct
from evdev import InputDevice, list_devices


def find_xbox360_controller():
    for path in list_devices():
        device = InputDevice(path)
        if "X-Box 360" in device.name:
            return device.path
    return None


class vJoy:
    def __init__(self, reference=1):
        self.reference = reference
        self.device = None
        self.acquired = False
        self.event_path = find_xbox360_controller()
        self.button_a_pressed = False

    def open(self):
        try:
            self.device = open(self.event_path, "wb")
            self.acquired = True
            return True
        except Exception as e:
            print("Failed to open vJoy device: {}".format(e))
            return False

    def close(self):
        try:
            if self.device:
                self.device.close()
            self.acquired = False
            return True
        except Exception as e:
            print("Failed to close vJoy device: {}".format(e))
            return False

    def generateJoystickPosition(
        self,
        wThrottle=0,
        wRudder=0,
        wAileron=0,
        wAxisX=0,
        wAxisY=0,
        wAxisZ=0,
        wAxisXRot=0,
        wAxisYRot=0,
        wAxisZRot=0,
        wSlider=0,
        wDial=0,
        wWheel=0,
        wAxisVX=0,
        wAxisVY=0,
        wAxisVZ=0,
        wAxisVBRX=0,
        wAxisVBRY=0,
        wAxisVBRZ=0,
        lButtons=0,
        bHats=0,
        bHatsEx1=0,
        bHatsEx2=0,
        bHatsEx3=0,
    ):
        joy_pos_format = "BlllllllllllllllllllIIII"
        return struct.pack(
            joy_pos_format,
            self.reference,
            wThrottle,
            wRudder,
            wAileron,
            wAxisX,
            wAxisY,
            wAxisZ,
            wAxisXRot,
            wAxisYRot,
            wAxisZRot,
            wSlider,
            wDial,
            wWheel,
            wAxisVX,
            wAxisVY,
            wAxisVZ,
            wAxisVBRX,
            wAxisVBRY,
            wAxisVBRZ,
            lButtons,
            bHats,
            bHatsEx1,
            bHatsEx2,
            bHatsEx3,
        )

    def _send_event(self, event_type, code, value):
        event = struct.pack("llHHi", 0, 0, event_type, code, value)
        self.device.write(event)
        self.device.flush()

    def update(self, joystickPosition):
        if not self.device or not self.acquired:
            return False

        try:
            values = struct.unpack("BlllllllllllllllllllIIII", joystickPosition)

            ev_abs = 0x03
            ev_key = 0x01
            abs_x = 0x00
            abs_z = 0x02
            abs_rz = 0x05
            btn_south = 0x130

            steer_value = values[4]
            scaled_steer = int(((steer_value / 32768) * 2 - 1) * 32767)
            scaled_steer = max(-32768, min(32767, scaled_steer))
            self._send_event(ev_abs, abs_x, scaled_steer)

            throttle_value = int(max(0, min(255, (values[5] / 32768.0) * 255)))
            self._send_event(ev_abs, abs_rz, throttle_value)

            brake_value = int(max(0, min(255, (values[6] / 32768.0) * 255)))
            self._send_event(ev_abs, abs_z, brake_value)

            button_a = bool(values[19] & 0x00000001)
            if button_a != self.button_a_pressed:
                self._send_event(ev_key, btn_south, 1 if button_a else 0)
                self.button_a_pressed = button_a

            self._send_event(0, 0, 0)
            return True
        except Exception as e:
            print("Failed to update vJoy device: {}".format(e))
            return False
