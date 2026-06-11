/*
 * arduino_drive_serial.ino
 * ------------------------
 * Receives differential drive commands from Jetson over Serial and
 * drives two BTS7960 H-bridge modules (left and right side).
 *
 * Protocol (ASCII, newline-terminated):
 *   L:<left>,R:<right>\n
 *   e.g.  L:+0.85,R:-0.42\n
 *
 *   left / right in [-1.0, +1.0]
 *     positive → forward  (RPWM active)
 *     negative → reverse  (LPWM active)
 *     0.0      → coast    (both PWM = 0)
 *
 * BTS7960 wiring (one module per side):
 *   RPWM → forward PWM input
 *   LPWM → reverse PWM input
 *   R_EN, L_EN → tie HIGH (or drive HIGH from a digital pin to enable)
 *   VCC  → 5 V logic
 *   B+   → motor supply (up to 43 V)
 *
 * Pin assignments (change to suit your wiring):
 */

// ── Pin definitions ───────────────────────────────────────────────────────────

// Left side BTS7960
const int L_RPWM = 5;   // forward  (PWM pin)
const int L_LPWM = 6;   // reverse  (PWM pin)

// Right side BTS7960
const int R_RPWM = 9;   // forward  (PWM pin)
const int R_LPWM = 10;  // reverse  (PWM pin)

// ── Config ────────────────────────────────────────────────────────────────────

const long         BAUD       = 115200;
const int          MAX_PWM    = 255;    // Arduino analogWrite ceiling
const int          MIN_PWM    = 30;     // below this → treat as 0 (deadband)
const unsigned long TIMEOUT_MS = 1000;  // stop motors if no packet for 1 s (Jetson sends every 250 ms)

// ── Globals ───────────────────────────────────────────────────────────────────

char          rxBuf[64];
uint8_t       rxIdx = 0;
unsigned long lastPacketMs = 0;

// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Drive one BTS7960 channel.
 * @param rpwm   forward PWM pin
 * @param lpwm   reverse PWM pin
 * @param value  [-1.0, +1.0]
 */
void driveChannel(int rpwm, int lpwm, float value) {
    int pwm = (int)(abs(value) * MAX_PWM);
    if (pwm < MIN_PWM) pwm = 0;       // deadband

    if (pwm == 0 || value == 0.0f) {
        analogWrite(rpwm, 0);
        analogWrite(lpwm, 0);
    } else if (value > 0.0f) {        // forward
        analogWrite(rpwm, pwm);
        analogWrite(lpwm, 0);
    } else {                          // reverse
        analogWrite(rpwm, 0);
        analogWrite(lpwm, pwm);
    }
}

void stopAll() {
    driveChannel(L_RPWM, L_LPWM, 0.0f);
    driveChannel(R_RPWM, R_LPWM, 0.0f);
}

/**
 * Parse "L:+0.85,R:-0.42" and drive both channels.
 * Returns true on successful parse.
 */
bool parseAndDrive(const char* line) {
    float left = 0.0f, right = 0.0f;

    const char* lPtr = strstr(line, "L:");
    const char* rPtr = strstr(line, "R:");
    if (!lPtr || !rPtr) return false;

    left  = atof(lPtr + 2);
    right = atof(rPtr + 2);

    // Clamp to [-1, +1]
    left  = max(-1.0f, min(1.0f, left));
    right = max(-1.0f, min(1.0f, right));

    driveChannel(L_RPWM, L_LPWM, left);
    driveChannel(R_RPWM, R_LPWM, right);

    // Debug echo
    Serial.print("[rx] L=");
    Serial.print(left, 3);
    Serial.print("  R=");
    Serial.println(right, 3);

    return true;
}

// ── Setup / Loop ──────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(BAUD);

    pinMode(L_RPWM, OUTPUT); pinMode(L_LPWM, OUTPUT);
    pinMode(R_RPWM, OUTPUT); pinMode(R_LPWM, OUTPUT);

    stopAll();
    lastPacketMs = millis();

    Serial.println("[arduino] drive receiver ready");
}

void loop() {
    // ── Read serial bytes until newline ──────────────────────────────────────
    while (Serial.available()) {
        char c = (char)Serial.read();

        if (c == '\n' || c == '\r') {
            if (rxIdx > 0) {
                rxBuf[rxIdx] = '\0';
                if (parseAndDrive(rxBuf)) {
                    lastPacketMs = millis();
                }
                rxIdx = 0;
            }
        } else {
            if (rxIdx < sizeof(rxBuf) - 1) {
                rxBuf[rxIdx++] = c;
            } else {
                rxIdx = 0;   // overflow — reset
            }
        }
    }

    // ── Watchdog: stop if Jetson goes silent ──────────────────────────────────
    if (millis() - lastPacketMs > TIMEOUT_MS) {
        stopAll();
    }
}
