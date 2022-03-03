#include <MPU6050.h>
#include <Wire.h>
#include <SPI.h>
#include <nRF24L01.h>
#include <RF24.h>
MPU6050 sensor;

int ivmeX , ivmeY , ivmeZ , GyroX , GyroY , GyroZ;

RF24 radio(9, 10); // CE, CSN
const byte address[6] = "ADU";
void setup() {
  radio.begin();
  Wire.begin();
  sensor.initialize();
  radio.openWritingPipe(address);
  radio.setPALevel(RF24_PA_MIN);
  radio.stopListening();
}
void loop() {
 sensor.getAcceleration(&ivmeX, &ivmeY, &ivmeZ);
  sensor.getRotation(&GyroX, &GyroY, &GyroY);
  typedef struct Value {
              float x;
              float y;
              float z;
              float e0;
              float e1;
              float e2;   
            } Value;

            Value value;

            value.x=ivmeX;
            value.y=ivmeY;
            value.z=ivmeZ;
            value.e0=GyroX;
            value.e1=GyroY;
            value.e2=GyroY;

radio.write(&value, sizeof(value));
}
