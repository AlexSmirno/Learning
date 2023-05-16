package com.laba.rabbitmqlaba;

import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

@Component
public class Runner implements CommandLineRunner {
    private final RabbitTemplate rabbitTemplate;

    public Runner(RabbitTemplate rabbitTemplate) {
        this.rabbitTemplate = rabbitTemplate;
    }

    @Override
    public void run(String... args) throws Exception {
        while(true) {
            System.out.println("Sending message...");
            rabbitTemplate.convertAndSend(RabbitMqLabaApplication.exchangeName,
                    RabbitMqLabaApplication.queueName,
                    "Hello from " +
                    RabbitMqLabaApplication.exchangeName);
            Thread.sleep(2000);
        }
    }
}
