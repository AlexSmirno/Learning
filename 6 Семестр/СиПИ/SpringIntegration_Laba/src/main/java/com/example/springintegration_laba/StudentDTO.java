package com.example.springintegration_laba;

import java.util.Date;
import java.util.Random;

public class StudentDTO {
    public String fullName;
    public String birthDay;
    public int randomValue;
    public String getFullName()
    {
        return fullName;
    }
    public String getBirthDay()
    {
        return birthDay;
    }
    public int getRandomValue(){
        return randomValue;
    }
    public String toString() { return getFullName() + " " + getBirthDay() + " randomValue = " + randomValue; }

    public StudentDTO setFullName(String name)
    {
        fullName = name;
        return this;
    }
    public StudentDTO setBirthDay(String date)
    {
        birthDay = date;
        return this;
    }
    public StudentDTO setRandomValue()
    {
        Random random = new Random();
        randomValue = random.nextInt(99) + 1;
        return this;
    }
}
