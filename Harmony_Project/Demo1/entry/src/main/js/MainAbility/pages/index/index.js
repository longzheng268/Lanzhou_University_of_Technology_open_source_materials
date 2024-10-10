import sensor from '@system.sensor';
export default {
    data: {
        steps: 0,
        heartRate: 0
    },
    onInit(){
        var that = this//回调函数中的 this 不是变量 steps 所在的 this 对象
        sensor.subscribeStepCounter({//订阅计步传感器
            success: function(ret) {//步数改变时的回调函数
                that.steps = ret.steps;
            },
            fail: function(data, code) {//订阅失败时的回调函数
                console.log('Subscription failed. Code: ' + code + '; Data: ' + data);
            },
        });
        sensor.subscribeHeartRate({//订阅心率传感器
            success: function(ret) {//心率改变时的回调函数
                that.heartRate = ret.heartRate;
            },
            fail: function(data, code) {//订阅失败时的回调函数
                console.log('Subscription failed. Code: ' + code + '; Data: ' + data);
            },
        });
    },
    onDestroy(){
        sensor.unsubscribeStepCounter();//取消订阅计步传感器
        sensor.unsubscribeHeartRate();//取消订阅心率传感器
    }
}