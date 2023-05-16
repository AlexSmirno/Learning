package com.example.springintegration_laba;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.messaging.MessageChannel;
import org.springframework.messaging.support.MessageBuilder;
import org.springframework.stereotype.Component;

@Component
public class HelloWorldGateway {

    private final MessageChannel messageChannel;

    @Autowired
    public HelloWorldGateway(@Qualifier("filteredChannel") MessageChannel messageChannel) {
        this.messageChannel = messageChannel;
    }

    public void send(StudentDTO event) {
        messageChannel.send(MessageBuilder.withPayload(event).build());
    }
}
