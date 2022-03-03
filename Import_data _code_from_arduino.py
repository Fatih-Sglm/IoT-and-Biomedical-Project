# -*- coding: utf-8 -*-
import serial,datetime
import csv
ser=serial.Serial('COM4',9600 , timeout= 1 )
ser.close()
ser.open()
ser.flushInput()
if ser.isOpen():
    while True:
        inline = str(ser.readline().strip())
        time = datetime.datetime.now()
        ct = time.strftime("%H:%M:%S")
        inline=inline.replace("'","")
        inline=inline.replace("b","")
        try:
            info=inline.split(",")
            info2=[ct,float(info[0]), float(info[1]),float(info[2])]
            print(info2)
            with open("test_data.csv","a",newline='') as f:
              writer = csv.writer(f,delimiter=",")
              writer.writerow(info2)
              f.close()
        except:
            print("Okuma hatası")
    
