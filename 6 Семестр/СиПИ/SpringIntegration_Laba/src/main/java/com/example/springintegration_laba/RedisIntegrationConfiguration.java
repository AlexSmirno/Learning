package com.example.springintegration_laba;

import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.data.redis.serializer.Jackson2JsonRedisSerializer;
import org.springframework.integration.annotation.Filter;
import org.springframework.integration.channel.PublishSubscribeChannel;
import org.springframework.integration.config.EnableIntegration;
import org.springframework.integration.dsl.IntegrationFlow;
import org.springframework.integration.dsl.IntegrationFlows;
import org.springframework.integration.redis.inbound.RedisQueueMessageDrivenEndpoint;
import org.springframework.integration.redis.outbound.RedisQueueOutboundChannelAdapter;
import org.springframework.messaging.Message;
import org.springframework.messaging.MessageChannel;
import org.springframework.messaging.SubscribableChannel;

@Configuration
@EnableIntegration
public class RedisIntegrationConfiguration {
    private static final String QUEUE_HELLO_WORLD = "queue:hello-world";

    @Bean("helloWorldInboundChannelFlow")
    public IntegrationFlow redisHelloWorldEventInboundChannelFlow(
            RedisConnectionFactory redisConnectionFactory,
            @Qualifier("helloWorldInboundChannel") MessageChannel channel
    ) {
        RedisQueueMessageDrivenEndpoint endpoint =
                new RedisQueueMessageDrivenEndpoint(QUEUE_HELLO_WORLD, redisConnectionFactory);
        Jackson2JsonRedisSerializer<StudentDTO> serializer
                = new Jackson2JsonRedisSerializer<>(StudentDTO.class);

        endpoint.setSerializer(serializer);
        endpoint.setBeanName("helloWorldRedisQueueMessageDrivenEndpoint");

        return IntegrationFlows
                .from(endpoint)
                .channel(channel)
                .get();
    }

    @Bean("helloWorldOutboundChannelFlow")
    public IntegrationFlow redisHelloWorldEventOutboundChannelFlow(
            RedisConnectionFactory redisConnectionFactory,
            @Qualifier("helloWorldOutboundChannel") MessageChannel channel
    ) {
        Jackson2JsonRedisSerializer<StudentDTO> serializer
                = new Jackson2JsonRedisSerializer<>(StudentDTO.class);

        RedisQueueOutboundChannelAdapter channelAdapter =
                new RedisQueueOutboundChannelAdapter(QUEUE_HELLO_WORLD, redisConnectionFactory);
        channelAdapter.setSerializer(serializer);
        return IntegrationFlows
                .from(channel)
                .handle(channelAdapter)
                .get();
    }

    @Bean("helloWorldOutboundChannel")
    public SubscribableChannel logEventOutboundChannel() {
        PublishSubscribeChannel channel = new PublishSubscribeChannel();
        channel.setComponentName("helloWorldOutboundChannel");

        return channel;
    }

    @Bean("helloWorldInboundChannel")
    public SubscribableChannel logEventInboundChannel() {
        PublishSubscribeChannel channel = new PublishSubscribeChannel();
        channel.setComponentName("helloWorldInboundChannel");
        return channel;
    }

    @Bean("filteredChannel")
    public SubscribableChannel filteredChannel() {
        PublishSubscribeChannel channel = new PublishSubscribeChannel();
        channel.setComponentName("filteredChannel");
        return channel;
    }

    //filteredChannel
    @Filter (inputChannel = "filteredChannel", outputChannel = "helloWorldOutboundChannel")
    boolean filter(Message<StudentDTO> message) {
        return (message.getPayload().getRandomValue() & 1) == 1;
    }
}
