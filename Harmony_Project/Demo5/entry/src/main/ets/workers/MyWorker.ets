import { ErrorEvent, MessageEvents, ThreadWorkerGlobalScope, worker }
  from '@kit.ArkTS';
//下面是我们增加的导包
import CommonEventManager from '@ohos.commonEventManager';//公共事件管理
import { atomicService } from '@kit.ScenarioFusionKit';//获取系统设置
import { BusinessError } from '@kit.BasicServicesKit';//错误信息


//这一部分是开发工具自动生成的
const workerPort: ThreadWorkerGlobalScope = worker.workerPort;

/**
 * Defines the event handler to be called when the worker thread receives a message sent
 by the host thread.
 * The event handler is executed in the worker thread.
 *
 * @param e message data
 */
workerPort.onmessage = (e: MessageEvents) => {
}

/**
 * Defines the event handler to be called when the worker receives a message that cannot
 be deserialized.
 * The event handler is executed in the worker thread.
 *
 * @param e message data
 */
workerPort.onmessageerror = (e: MessageEvents) => {
}

/**
 * Defines the event handler to be called when an exception occurs during worker
 execution.
 * The event handler is executed in the worker thread.
 *
 * @param e error message
 */
workerPort.onerror = (e: ErrorEvent) => {
}

//下面部分是我们增加的代码
let syswifienabled: boolean | undefined = undefined //wifi 状态
let syslocationenabled: boolean | undefined = undefined //位置信息（定位开关）

//获取系统信息，若改变则发布公共事件通知订阅者
function monitor()
{

  let stateArray: Array<atomicService.SystemSettingType> =
    [ 'locationEnabled', 'wifiEnabled']
  try {
    let data = atomicService.getSystemSetting(stateArray)
    console.info('succeeded in getting system setting info');
    //公共事件相关信息
    let options: CommonEventManager.CommonEventPublishData = {
      code: 0, //公共事件的数据代码
      data: "", //公共事件的数据内容
      isOrdered: true //有序公共事件
    }
    if (data.wifiEnabled !== syswifienabled) { //wifi 状态改变
      syswifienabled = data.wifiEnabled
      options.code = 1 //用 1 表示 wifi 状态改变
      if (data.wifiEnabled) {
        options.data = "open" //wifi 打开
      } else {
        options.data = "close" //wifi 关闭
      }
      try {
        //发布自定义公共事件 usual.event.SYS_STATE_CHANGED（自定义名字）
        CommonEventManager.publish("usual.event.SYS_STATE_CHANGED", options,
          (err: BusinessError) => {
            if (err) {
              //注意'和`的区别，`括起来的串中含有变量
              console.error(`publish wifi_changed failed, code is ${err.code}, message
is ${err.message}`);
            } else {
              console.info("publish wifi_changed sucessed");
            }
          });
      } catch (error) {
        let err: BusinessError = error as BusinessError;
        console.error(`publish wifi_changed failed, code is ${err.code}, message is
${err.message}`);
      }
    }
    if (data.locationEnabled !== syslocationenabled) { //位置信息（定位开关）改变
      syslocationenabled = data.locationEnabled
      options.code = 2 //用 2 表示位置信息（定位开关）改变 信息技术与工程开放实验 B
      35

      if (data.locationEnabled) {
        options.data = "open" //位置信息（定位开关）打开
      } else {
        options.data = "close" //位置信息（定位开关）关闭
      }
      try {
        //发布自定义公共事件 usual.event.SYS_STATE_CHANGED（自定义名字）
        CommonEventManager.publish("usual.event.SYS_STATE_CHANGED", options,
          (err: BusinessError) => {
            if (err) {
              //注意'和`的区别，`括起来的串中含有变量
              console.error(`publish location_changed failed, code is ${err.code},
message is ${err.message}`);
            } else {
              console.info("publish location_changed sucessed");
            }
          });
      } catch (error) {
        let err: BusinessError = error as BusinessError;
        console.error(`publish location_changed failed, code is ${err.code}, message
is ${err.message}`);
      }
    }
  } catch (error) {
    console.error('failReason: %{public}d %{public}s', error.code, error.message);
  }
}

setInterval(monitor,1000)//每 1000ms 执行一次函数