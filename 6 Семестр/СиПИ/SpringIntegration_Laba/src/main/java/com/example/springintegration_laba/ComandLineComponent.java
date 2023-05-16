package com.example.springintegration_laba;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

import java.util.Date;

@Component
public class ComandLineComponent implements CommandLineRunner {
    private final HelloWorldGateway helloWorldGateway;

    @Autowired
    public ComandLineComponent(HelloWorldGateway helloWorldGateway) {
        this.helloWorldGateway = helloWorldGateway;
    }

    @Override
    public void run(String... args) throws Exception {
        System.out.println("Messages to send");

        for (int i = 0; i < 10; i++){
            StudentDTO dto = new StudentDTO();
            dto.setFullName("Smirnov AA").setBirthDay("08.07.2002").setRandomValue();
            System.out.println(dto.toString());
            helloWorldGateway.send(dto);
        }
    }
}
