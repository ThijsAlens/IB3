#define Mpin1 6

void setup(){
    //Serial.begin(9600);
    pinMode(Mpin1, OUTPUT);
}

void loop(){

  String data;

  if (Serial.available()) {
    delay(10);
    while (Serial.available() > 0) {
      data += (char)Serial.read();
    }
    Serial.flush();
    Serial.println(data);
  }
    
    
    for (int i = 0; i < 256; i++){
        analogWrite(Mpin1, i);
        delay(50);
    }
}
