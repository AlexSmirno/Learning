package com.example.springintegration_laba;

import org.springframework.integration.annotation.ServiceActivator;
import org.springframework.messaging.Message;
import org.springframework.stereotype.Component;

@Component
public class HelloWorldActivator {
    @ServiceActivator(inputChannel = "helloWorldInboundChannel")
    public void activate(Message<StudentDTO> event) {
        System.out.println(event.getPayload().toString());
    }
}
