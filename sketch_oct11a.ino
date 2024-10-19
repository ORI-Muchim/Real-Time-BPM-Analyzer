const int RELAY_PIN = 7;  // 릴레이 모듈이 연결된 디지털 핀

void setup() {
  Serial.begin(9600);
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    if (command == '1') {
      digitalWrite(RELAY_PIN, HIGH);  // 릴레이 ON
    } else if (command == '0') {
      digitalWrite(RELAY_PIN, LOW);   // 릴레이 OFF
    }
  }
}
