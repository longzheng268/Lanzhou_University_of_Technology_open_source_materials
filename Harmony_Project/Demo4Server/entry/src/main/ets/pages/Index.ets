import { socket } from '@kit.NetworkKit';//网络管理 
import { BusinessError } from '@kit.BasicServicesKit';//基础服务 
import utils from '../utils/utils'//自定义公用函数 
@Entry
@Component
struct Index {
  @State serverip: string = ''
  @State mess: string = ""//保存客户端发来的信息
  // 创建一个 TCPSocketServer 连接，返回一个 TCPSocketServer 对象。
  private tcpServer: socket.TCPSocketServer
    = socket.constructTCPSocketServerInstance();
  private client: socket.TCPSocketConnection
    = {} as socket.TCPSocketConnection//客户端
  private serverbegined: boolean = false//服务器开始服务

  aboutToAppear(): void {
    //获得本机 wifi 的 IP
    this.serverip=utils.getIp()
  }

  aboutToDisappear(): void {
    if (this.serverbegined) {
      //取消相关的事件订阅
      this.serverend()
    }
  }

  build() {
    Column() {
      Text("当前设备 IP：" + this.serverip)
        .fontSize(20)
        .fontWeight(FontWeight.Bold)
      Button("开始服务").margin({top:10,bottom:10}).width("50%")
        .onClick((event) => {
          if (!this.serverbegined) {
            this.serverstart()
            this.serverbegined = true
          }
        })
      Text("收到的消息："+this.mess)
    }
    .height('100%')
    .width('100%')
  }

  serverstart() {
    // 服务器地址（本机 IP 地址和端口）
    let ipAddress: socket.NetAddress = {} as socket.NetAddress;
    ipAddress.address = this.serverip;
    ipAddress.port = 4651;
    // 绑定到本机 IP 地址和端口并进行监听
    this.tcpServer.listen(ipAddress).then(() => {
      console.log('listen success');
    }).catch((err: BusinessError) => {
      console.log('listen fail');
    });

    //客户端信息（包含数据和 Ip、端口等）
    class SocketInfo {
      message: ArrayBuffer = new ArrayBuffer(1);
      remoteInfo: socket.SocketRemoteInfo = {} as socket.SocketRemoteInfo;
    }
    // 订阅 TCPSocketServer 的 connect 事件
    this.tcpServer.on("connect", (socketclient: socket.TCPSocketConnection) => {
      this.client=socketclient
      // 订阅客户端的 close 事件
      socketclient.on("close", () => {
        console.log("on close success");
      });
      // 订阅客户端的 message 事件
      socketclient.on("message", (value: SocketInfo) => {
        let buffer = value.message;//客户端发来的数据
        let str=utils.bufferToString(buffer)//转换为字符串
        console.log("received message--:" + str);

        this.mess = str//显示到界面

        // 发送函数的参数
        let tcpSendOptions: socket.TCPSendOptions = {} as socket.TCPSendOptions;
        tcpSendOptions.data = 'Hello, client!';//要发送的数据
        // 向客户端发送数据
        socketclient.send(tcpSendOptions).then(() => {
          console.log('send success');
        }).catch((err: Object) => {
          console.error('send fail: ' + JSON.stringify(err));
        });

        // 关闭与客户端的连接
        socketclient.close().then(() => {
          console.log('close success');
        }).catch((err: BusinessError) => {
          console.log('close fail');
        });
      });
    })
  };

  serverend() {
    // 取消客户端相关的事件订阅
    this.client.off("message");
    this.client.off("close");
    // 取消服务端相关的事件订阅
    this.tcpServer.off("connect");
  }
} 