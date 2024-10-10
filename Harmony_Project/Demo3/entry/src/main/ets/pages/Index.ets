import { geoLocationManager } from '@kit.LocationKit';//位置服务 
// AbilityKit 为 Ability 工具 
import { abilityAccessCtrl, common, Permissions } from '@kit.AbilityKit';
import { BusinessError } from '@kit.BasicServicesKit';//基础服务 

@Entry
@Component
struct Index {
  @State latitude:number=0//维度
  @State longitude:number=0//经度
  @State mess:string=""//提示信息

  //位置改变后的回调函数
  locationCallback = (location: geoLocationManager.Location): void => {
    console.log('locationCallback: data: ' + JSON.stringify(location));
    //location 为获得的位置
    this.latitude = location.latitude//维度
    this.longitude = location.longitude//经度

    //调用地理逆编码服务（经纬度转地址）所需的信息
    let reverseGeocodeRequest: geoLocationManager.ReverseGeoCodeRequest = {
      'locale': 'zh',//指定位置描述信息的语言，“zh”代表中文，“en”代表英文
      'latitude': this.latitude,//维度
      'longitude': this.longitude,//经度
      'maxItems': 1//指定返回位置信息的最大个数，默认值是 1
    };

    //调用地理逆编码服务，将坐标转换为地理描述
    //本地模拟器用 macOS(ARM)才支持地理逆编码服务，win10,win11 的模拟器不支持地理逆编码服务
    geoLocationManager.getAddressesFromLocation(reverseGeocodeRequest).then(data => {
      //data 中包含地址信息
      if (data[0].placeName) {
        this.mess = "地址为："+data[0].placeName;
      }
    }).catch((err: Error) => {
      console.error('GetAddressesFromLocation err ' + JSON.stringify(err));this.mess = "本地模拟器（X86）不支持地理逆编码服务。"
    });
  };

  Locationon() {
    //持续定位。多用于导航、运动轨迹、出行等场景。
    //request 包含要提供何种类型的位置服务，以及位置结果上报的频率
    let request: geoLocationManager.ContinuousLocationRequest = {
      'interval': 1,//表示上报位置信息的时间间隔，单位是秒，默认值为 1 秒
      'locationScenario': geoLocationManager.UserActivityScenario.NAVIGATION//导航场景
    }
    //监控位置改变，如改变则调用回调函数 this.locationCallback
    try {
      geoLocationManager.on('locationChange', request, this.locationCallback);
    } catch (err) {
      console.error("errCode:" + JSON.stringify(err));
      if (err.code=="3301100") {
        this.mess="请先开启手机的位置信息开关！"
      }
    }
  }
  //向用户申请权限
  reqPermissionsFromUser(permissions: Array<Permissions>,
    context: common.UIAbilityContext): void {
    let atManager: abilityAccessCtrl.AtManager =
      abilityAccessCtrl.createAtManager();//Ability 管理
    // requestPermissionsFromUser 会判断权限的授权状态来决定是否唤起弹窗
    atManager.requestPermissionsFromUser(context, permissions).then((data) => {
      //data 为授权结果
      let grantStatus: Array<number> = data.authResults;//授权结果数组
      let length: number = grantStatus.length;
      for (let i = 0; i < length; i++) {
        if (grantStatus[i] === 0) {
          // 用户授权，可以继续判断下一个权限
        } else {
          this.mess="用户拒绝授权，请在系统设置中为本应用开启位置权限！"
          return;
        }
      }
  // 授权成功
      this.Locationon();//监控位置改变并显示
    }).catch((err: BusinessError) => {
      console.error(`Failed to request permissions from user. Code is ${err.code},
message is ${err.message}`);
    })
  }

  aboutToAppear():void {
    //程序上下文
    let context: common.UIAbilityContext =
      getContext(this) as common.UIAbilityContext;
    //定位和模糊定位权限
    let permissions: Array<Permissions> =
      ["ohos.permission.LOCATION","ohos.permission.APPROXIMATELY_LOCATION"];
    //向用户申请权限
    this.reqPermissionsFromUser(permissions,context)
  }

  aboutToDisappear(): void {
    //关闭位置变化订阅，并删除对应的定位请求
    geoLocationManager.off('locationChange',this.locationCallback)
  }
  build() {
    Column() {
      Text("纬度："+this.latitude.toString())
        .fontSize(30).margin({top:100})
      Text("经度："+this.longitude.toString())
        .id('longitude')
        .fontSize(30)
      Text(this.mess)
        .id('mess')
        .fontSize(20)
    }
    .height('100%')
    .width('100%')
  }
}
