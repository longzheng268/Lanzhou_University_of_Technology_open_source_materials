if (!("finalizeConstruction" in ViewPU.prototype)) {
    Reflect.set(ViewPU.prototype, "finalizeConstruction", () => { });
}
interface Index_Params {
    serverip?: string;
    mess?: string;
    tcpServer?: socket.TCPSocketServer;
    client?: socket.TCPSocketConnection;
    serverbegined?: boolean;
}
import socket from "@ohos:net.socket";
import type { BusinessError } from "@ohos:base";
import utils from "@normalized:N&&&entry/src/main/ets/utils/utils&";
class Index extends ViewPU {
    constructor(parent, params, __localStorage, elmtId = -1, paramsLambda = undefined, extraInfo) {
        super(parent, __localStorage, elmtId, extraInfo);
        if (typeof paramsLambda === "function") {
            this.paramsGenerator_ = paramsLambda;
        }
        this.__serverip = new ObservedPropertySimplePU('', this, "serverip");
        this.__mess = new ObservedPropertySimplePU("" //保存客户端发来的信息
        // 创建一个 TCPSocketServer 连接，返回一个 TCPSocketServer 对象。
        , this, "mess");
        this.tcpServer = socket.constructTCPSocketServerInstance();
        this.client = {} as socket.TCPSocketConnection //客户端
        ;
        this.serverbegined = false //服务器开始服务
        ;
        this.setInitiallyProvidedValue(params);
        this.finalizeConstruction();
    }
    setInitiallyProvidedValue(params: Index_Params) {
        if (params.serverip !== undefined) {
            this.serverip = params.serverip;
        }
        if (params.mess !== undefined) {
            this.mess = params.mess;
        }
        if (params.tcpServer !== undefined) {
            this.tcpServer = params.tcpServer;
        }
        if (params.client !== undefined) {
            this.client = params.client;
        }
        if (params.serverbegined !== undefined) {
            this.serverbegined = params.serverbegined;
        }
    }
    updateStateVars(params: Index_Params) {
    }
    purgeVariableDependenciesOnElmtId(rmElmtId) {
        this.__serverip.purgeDependencyOnElmtId(rmElmtId);
        this.__mess.purgeDependencyOnElmtId(rmElmtId);
    }
    aboutToBeDeleted() {
        this.__serverip.aboutToBeDeleted();
        this.__mess.aboutToBeDeleted();
        SubscriberManager.Get().delete(this.id__());
        this.aboutToBeDeletedInternal();
    }
    private __serverip: ObservedPropertySimplePU<string>;
    get serverip() {
        return this.__serverip.get();
    }
    set serverip(newValue: string) {
        this.__serverip.set(newValue);
    }
    private __mess: ObservedPropertySimplePU<string>; //保存客户端发来的信息
    get mess() {
        return this.__mess.get();
    }
    set mess(newValue: string) {
        this.__mess.set(newValue);
    }
    // 创建一个 TCPSocketServer 连接，返回一个 TCPSocketServer 对象。
    private tcpServer: socket.TCPSocketServer;
    private client: socket.TCPSocketConnection; //客户端
    private serverbegined: boolean; //服务器开始服务
    aboutToAppear(): void {
        //获得本机 wifi 的 IP
        this.serverip = utils.getIp();
    }
    aboutToDisappear(): void {
        if (this.serverbegined) {
            //取消相关的事件订阅
            this.serverend();
        }
    }
    initialRender() {
        this.observeComponentCreation2((elmtId, isInitialRender) => {
            Column.create();
            Column.height('100%');
            Column.width('100%');
        }, Column);
        this.observeComponentCreation2((elmtId, isInitialRender) => {
            Text.create("当前设备 IP：" + this.serverip);
            Text.fontSize(20);
            Text.fontWeight(FontWeight.Bold);
        }, Text);
        Text.pop();
        this.observeComponentCreation2((elmtId, isInitialRender) => {
            Button.createWithLabel("开始服务");
            Button.margin({ top: 10, bottom: 10 });
            Button.width("50%");
            Button.onClick((event) => {
                if (!this.serverbegined) {
                    this.serverstart();
                    this.serverbegined = true;
                }
            });
        }, Button);
        Button.pop();
        this.observeComponentCreation2((elmtId, isInitialRender) => {
            Text.create("收到的消息：" + this.mess);
        }, Text);
        Text.pop();
        Column.pop();
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
            this.client = socketclient;
            // 订阅客户端的 close 事件
            socketclient.on("close", () => {
                console.log("on close success");
            });
            // 订阅客户端的 message 事件
            socketclient.on("message", (value: SocketInfo) => {
                let buffer = value.message; //客户端发来的数据
                let str = utils.bufferToString(buffer); //转换为字符串
                console.log("received message--:" + str);
                this.mess = str; //显示到界面
                // 发送函数的参数
                let tcpSendOptions: socket.TCPSendOptions = {} as socket.TCPSendOptions;
                tcpSendOptions.data = 'Hello, client!'; //要发送的数据
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
        });
    }
    serverend() {
        // 取消客户端相关的事件订阅
        this.client.off("message");
        this.client.off("close");
        // 取消服务端相关的事件订阅
        this.tcpServer.off("connect");
    }
    rerender() {
        this.updateDirtyElements();
    }
    static getEntryName(): string {
        return "Index";
    }
}
registerNamedRoute(() => new Index(undefined, {}), "", { bundleName: "com.example.demo4server", moduleName: "entry", pagePath: "pages/Index", pageFullPath: "entry/src/main/ets/pages/Index", integratedHsp: "false" });
