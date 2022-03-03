#include <SPI.h>
#include <nRF24L01.h>
#include <RF24.h>
RF24 radio(9, 10); // CE, CSN
const byte address[10] = "ADU";
void setup() {
  Serial.begin(9600);
  radio.begin();
  radio.openReadingPipe(0, address);
  radio.setPALevel(RF24_PA_MIN);
  radio.startListening();
}
void loop() {
  if (radio.available()){
     typedef struct Value {
              float x;
              float y;
              float z;             
              float g1;
              float g2;
              float g3; 
            } Value;

            Value value;
            
    //radio.read(&text, sizeof(text));
    radio.read(&value, sizeof(value));
    Serial.print(value.x);
    Serial.print(",");
    Serial.print(value.y);
    Serial.print(",");
    Serial.print(value.z);
    Serial.print(",");
    Serial.print(value.g1);
    Serial.print(",");
    Serial.print(value.g2);
    Serial.print(",");
    Serial.println(value.g3);
  }
}
