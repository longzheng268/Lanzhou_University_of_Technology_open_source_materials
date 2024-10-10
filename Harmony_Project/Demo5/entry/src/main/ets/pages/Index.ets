import CommonEventManager from '@ohos.commonEventManager';//公共事件管理
import { worker } from '@kit.ArkTS';//多线程

// 使用 Woeker 执行并发程序，避免主线程执行长时间任务导致界面卡顿
//Woeker 中程序为每 1000ms 获取一次系统信息，如改变则发布公共事件通知订阅者
const workerInstance: worker.ThreadWorker =
  new worker.ThreadWorker('entry/ets/workers/MyWorker.ets');

@Entry
@Component
struct Index {
  @State wifi_status: string = ""//wifi 开关
  @State airplane_mode_status:string="close"//飞行模式开关（飞行模式暂无法获取初值）
  @State location_status: string = ""//位置信息（定位开关）

  // 订阅者, 用于保存创建成功的订阅者对象，后续使用其完成订阅及退订的动作
  private sysstatesubscriber: CommonEventManager.CommonEventSubscriber | undefined
    = undefined;//系统信息订阅者
  private airplane_modesubscriber: CommonEventManager.CommonEventSubscriber | undefined
    = undefined;//飞行模式订阅者

  //订阅公共事件
  createsubscriber(){
    console.info("subscribesyschanged");
    // 订阅信息
    let sysstatesubscribeInfo: CommonEventManager.CommonEventSubscribeInfo = {
      events: ['usual.event.SYS_STATE_CHANGED']//自定义公共事件名字
    };
    // 根据订阅信息（sysstatesubscribeInfo）创建订阅者
    CommonEventManager.createSubscriber(sysstatesubscribeInfo)
      .then((commonEventSubscriber) => {
        console.info("createsysstateSubscriber");
        this.sysstatesubscriber = commonEventSubscriber;//保存订阅者
        try {
          // 订阅公共事件（即收到公共事件时的处理）
          CommonEventManager.subscribe(this.sysstatesubscriber, (err, data) => {
            console.info('sysstatesubscribeCallBack, data is:' + JSON.stringify(data));
            if (err) {
              //注意'和`的区别，`括起来的串中含有变量
              console.error(`sysstatesubscribe failed, code is ${err?.code}, message is
${err?.message}`);
              return;
            }
            //data 为接收到的公共事件的数据
            let code=data.code as number//公共事件的数据代码
            let status=data.data as string//公共事件的数据内容
            if (code==1){//wifi 状态改变
              if (this.wifi_status!=status) this.wifi_status=status
            }else if(code==2){//位置信息（定位开关）改变
              if (this.location_status!=status) this.location_status=status
            }
          });
        } catch (error) {
          console.error( "sysstateSubscribe failed " + JSON.stringify(error));
        }
      });

    // 订阅信息
    let airplane_modesubscribeInfo: CommonEventManager.CommonEventSubscribeInfo = {

      //系统定义的公共事件名字，该公共事件由系统自动发布

      events: ['usual.event.AIRPLANE_MODE']
    };
    // 根据订阅信息（airplane_modesubscribeInfo）创建订阅者
    CommonEventManager.createSubscriber(airplane_modesubscribeInfo)
      .then((commonEventSubscriber) => {
        console.info("createairplane_modeSubscriber");
        this.airplane_modesubscriber = commonEventSubscriber;//保存订阅者
        try {
          // 订阅公共事件（即收到公共事件时的处理）
          CommonEventManager.subscribe(this.airplane_modesubscriber, (err, data) => {
            console.info('airplane_modesubscribeCallBack, data is:'
              + JSON.stringify(data));
            if (err) {
              //注意'和`的区别，`括起来的串中含有变量
              console.error(`airplane_modesubscribe failed, code is ${err?.code},
message is ${err?.message}`);
              return;
            }
            //data 为接收到的公共事件的数据
            let code=data.code as number//公共事件的数据代码
            //let status=data.data as string//公共事件的数据内容
            //this.airplane_mode_status=code.toString()+':'+status
            if (code==0){
              this.airplane_mode_status="close"
            }else if (code==1){
              this.airplane_mode_status="open"
            }

          });
        } catch (error) {
          console.error( "airplane_modeSubscribe failed " + JSON.stringify(error));
        }
      });
  }

  aboutToAppear(): void {
    //订阅公共事件
    this.createsubscriber();
  }

  aboutToDisappear(): void {
    //取消订阅公共事件
    if (this.sysstatesubscriber)
      CommonEventManager.unsubscribe(this.sysstatesubscriber);
    if (this.airplane_modesubscriber)
      CommonEventManager.unsubscribe(this.airplane_modesubscriber);
  }
  build() {
    Column() {
      Text("系统配置").fontSize(50).fontWeight(FontWeight.Bold).margin({ top: 100 })
      Text("无线网络：" + this.wifi_status).fontSize(30)
      Text("飞行模式：" + this.airplane_mode_status).fontSize(30)
      Text("位置开关：" + this.location_status).fontSize(30)
    }
    .height('100%')
    .width('100%')
  }
}