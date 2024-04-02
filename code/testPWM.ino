#define MAX_PWM 4

int pwmPins[] = {6, 5, 3, 2};
int pwmValue[] = {0, 0, 0, 0};
String resData[MAX_PWM] = {"0.00", "0.00", "0.00", "0.00"};
String data = "0.00";

void setup(){
    Serial.begin(9600);

    for (int i = 0; i < MAX_PWM; i++){
      pinMode(pwmPins[i], OUTPUT);
    }
}

void loop(){

  if (Serial.available()) {
    data = Serial.readStringUntil('\n');
    //Serial.flush();
    //Serial.println(data);
  }

  int i = 0;

  while (data.length() > 0 && i < MAX_PWM){
    int index = data.indexOf(',');
    if (index == -1){
      if (i == MAX_PWM-1){
        resData[i] = data;
      }
      break;
    } else{
      resData[i] = data.substring(0, index);
      data = data.substring(index+1);
    }

    ++i;

    /*
    Serial.print("resData[");
    Serial.print(i);
    Serial.print("]: ");
    Serial.println(resData[i]);

    Serial.print("pwmValue[");
    Serial.print(i);
    Serial.print("]: ");
    Serial.println(pwmValue[i]);

    Serial.print("restData: ");
    Serial.println(data);
    */
  }

  for (int j = 0; j < MAX_PWM; j++){
    //80 = threshold for if the motor needs to start
    pwmValue[j] = resData[j].toFloat() * (255-80) / 5 + 80; 
    if (pwmValue[j] == 80){
      pwmValue[j] = 0;
    }

    analogWrite(pwmPins[j], pwmValue[j]);
  }
}
