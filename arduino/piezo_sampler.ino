const int PIEZO_PIN = A0;

void setup() {
  Serial.begin(115200);
  pinMode(PIEZO_PIN, INPUT);
}

void loop() {
  int reading = analogRead(PIEZO_PIN);
  Serial.println(reading);
  delay(20);  // 50 Hz sampling
}
