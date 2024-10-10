import { sensor } from '@kit.SensorServiceKit';
import { BusinessError } from '@kit.BasicServicesKit';
@Entry
@Component
struct Index {
  @State rotateAngle:number=45;// Z 轴旋转角


  aboutToAppear(): void {
    try {
      //每 50ms 获取一次 Z 轴旋转角

      sensor.on(sensor.SensorId.ORIENTATION, (data: sensor.OrientationResponse) => {
        console.info('Succeeded in the device rotating at an angle around the Z axis:' + data.alpha);
        console.info('Succeeded in the device rotating at an angle around the X axis:' + data.beta);
        console.info('Succeeded in the device rotating at an angle around the Y axis:' + data.gamma);
        this.rotateAngle=data.alpha;
      }, { interval: 500000000 });//50ms
    } catch (error) {
      let e: BusinessError = error as BusinessError;
      console.error(`Failed to invoke on. Code: ${e.code}, message: ${e.message}`);
    }
  }
  aboutToDisappear(): void {
    //取消方向传感器的持续监听
    sensor.off(sensor.SensorId.ORIENTATION);
  }
  build() {
    Column() {
      Image($r('app.media.compass')).objectFit(ImageFit.Contain).margin(15)
        .rotate({ x: 0, y: 0, z: 1,centerX: '50%',centerY: '50%',
          angle: this.rotateAngle })//在图片中心点绕 Z 轴旋转 angle 度
    }
    .height('100%')
    .width('100%')
  }
}
