import { socket } from '@kit.NetworkKit';//网络服务
import { BusinessError } from '@kit.BasicServicesKit';//基础服务
import { promptAction } from '@kit.ArkUI';//方舟 UI 框架
import utils from '../utils/utils'//自定义公用函数库
@Entry
@Component
struct Index {
  @State message: string = '';//保存服务器发来的信息
  @State inputMessage:string=""//保存输入的服务器 IP
  private isSend:boolean=false//是否已向服务器成功发送
  private serverip:string=''
  private clientip:string=''

  aboutToAppear(): void {
    //获得本机 IP
    this.clientip=utils.getIp()
  }

  build() {
    Column() {
      TextInput({text: this.inputMessage,placeholder: "请输入服务器 IP"})
        .width('85%').height(50).maxLength(15).fontSize(20).type(InputType.Normal).
      margin({bottom:20})
        .onChange((value: string) => {
          this.inputMessage = value;//保存输入的服务器 IP
        })
      Button("发送").width('50%').enabled(!this.isSend)//只成功发送一次信息
        .onClick((event)=>{
          this.serverip=this.inputMessage.trim()
          this.send()//向服务器发送信息
        })
      Text(this.message)//显示服务器发来的信息
        .fontSize(20).fontWeight(FontWeight.Bold).margin({top:20})

    }
    .height('100%')
    .width('100%')
  }

  send(){
    //socket 远端信息
    class SocketInfo {
      message: ArrayBuffer = new ArrayBuffer(1);//远端发来的消息
      remoteInfo: socket.SocketRemoteInfo = {} as socket.SocketRemoteInfo;//远端 ip地址、端口等
    }
    // 创建一个 TCPSocket 连接，返回一个 TCPSocket 对象。
    let tcp: socket.TCPSocket = socket.constructTCPSocketInstance();
    //订阅远端发来的信息
    tcp.on('message', (value: SocketInfo) => {
      console.log("on message");
      let buffer = value.message;//接收到的服务器发来的消息
      let str =utils.bufferToString(buffer)//把 ArrayBuffer 转换到 string
      console.log("on connect received:" + str);
      this.message="服务器发来的信息："+str//显示到界面
    });
    //订阅 connect
    tcp.on('connect', () => {
      console.log("on connect");
    });
    //订阅 close
    tcp.on('close', () => {
      console.log("on close");
    });

    // 本机 IP 地址和端口。
    let ipAddress : socket.NetAddress = {} as socket.NetAddress;
    ipAddress.address = this.clientip;
    ipAddress.port = 1234;
    //绑定本机 IP 地址和端口
    tcp.bind(ipAddress, (err: BusinessError) => {
      if (err) {
        console.log('bind fail');
        return;

      }
      console.log('bind success');

      // 远端的 IP 地址和端口
      ipAddress.address = this.serverip;
      ipAddress.port =4651;
      // 连接参数信息
      let tcpConnect : socket.TCPConnectOptions = {address:ipAddress,timeout:6000} as
      socket.TCPConnectOptions;
      // 连接到指定的 IP 地址和端口
      tcp.connect(tcpConnect).then(() => {
        console.log('client connect success');
        //发送参数信息
        let tcpSendOptions: socket.TCPSendOptions = {
          data: 'Hello, server!'
        }
        //发送信息
        tcp.send(tcpSendOptions).then(() => {
          this.isSend=true//成功发送
          console.log('client send success');
        }).catch((err: BusinessError) => {
          console.log('client send fail');
        });
      }).catch((err: BusinessError) => {
        promptAction.showToast({ message:"连接服务器失败，服务器 IP 不对或服务器没开始服务！",duration:5})
        console.log('client connect fail');
      });
    });

    // 连接使用完毕后，主动关闭并取消相关事件的订阅。
    setTimeout(() => {
      tcp.close().then(() => {
        console.log('client close success');
      }).catch((err: BusinessError) => {
        console.log('client close fail');
      });
      tcp.off('message');
      tcp.off('connect');
      tcp.off('close');

    }, 30 * 1000);//30 秒
  }
}