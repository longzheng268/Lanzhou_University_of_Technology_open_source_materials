import wifiManager from "@ohos:wifiManager";
export default class Utils {
    //根据数字形式的 IP 地址获取字符串形式的 IP 地址
    static getIpAddrFromNum(ipNum: number): string {
        return (ipNum >>> 24) + '.' + (ipNum >> 16 & 0xFF) + '.' + (ipNum >> 8 & 0xFF) +
            '.' + (ipNum & 0xFF);
    }
    //获得本机 wifi 的 IP 地址
    static getIp(): string {
        let ipInfo = wifiManager.getIpInfo();
        let ipAddr = Utils.getIpAddrFromNum(ipInfo.ipAddress);
        return ipAddr;
    }
    //把 buffer 中的内容转换成字符串
    static bufferToString(buffer: ArrayBuffer) {
        let dataView = new DataView(buffer);
        let str = "";
        for (let i = 0; i < dataView.byteLength; ++i) {
            str += String.fromCharCode(dataView.getUint8(i));
        }
        return str;
    }
}
