if (!("finalizeConstruction" in ViewPU.prototype)) {
    Reflect.set(ViewPU.prototype, "finalizeConstruction", () => { });
}
interface Index_Params {
    rotateAngle?: number;
}
import sensor from "@ohos:sensor";
import type { BusinessError } from "@ohos:base";
class Index extends ViewPU {
    constructor(parent, params, __localStorage, elmtId = -1, paramsLambda = undefined, extraInfo) {
        super(parent, __localStorage, elmtId, extraInfo);
        if (typeof paramsLambda === "function") {
            this.paramsGenerator_ = paramsLambda;
        }
        this.__rotateAngle = new ObservedPropertySimplePU(45, this, "rotateAngle");
        this.setInitiallyProvidedValue(params);
        this.finalizeConstruction();
    }
    setInitiallyProvidedValue(params: Index_Params) {
        if (params.rotateAngle !== undefined) {
            this.rotateAngle = params.rotateAngle;
        }
    }
    updateStateVars(params: Index_Params) {
    }
    purgeVariableDependenciesOnElmtId(rmElmtId) {
        this.__rotateAngle.purgeDependencyOnElmtId(rmElmtId);
    }
    aboutToBeDeleted() {
        this.__rotateAngle.aboutToBeDeleted();
        SubscriberManager.Get().delete(this.id__());
        this.aboutToBeDeletedInternal();
    }
    private __rotateAngle: ObservedPropertySimplePU<number>; // Z 轴旋转角
    get rotateAngle() {
        return this.__rotateAngle.get();
    }
    set rotateAngle(newValue: number) {
        this.__rotateAngle.set(newValue);
    }
    aboutToAppear(): void {
        try {
            //每 50ms 获取一次 Z 轴旋转角
            sensor.on(sensor.SensorId.ORIENTATION, (data: sensor.OrientationResponse) => {
                console.info('Succeeded in the device rotating at an angle around the Z axis:' + data.alpha);
                console.info('Succeeded in the device rotating at an angle around the X axis:' + data.beta);
                console.info('Succeeded in the device rotating at an angle around the Y axis:' + data.gamma);
                this.rotateAngle = data.alpha;
            }, { interval: 500000000 }); //50ms
        }
        catch (error) {
            let e: BusinessError = error as BusinessError;
            console.error(`Failed to invoke on. Code: ${e.code}, message: ${e.message}`);
        }
    }
    aboutToDisappear(): void {
        //取消方向传感器的持续监听
        sensor.off(sensor.SensorId.ORIENTATION);
    }
    initialRender() {
        this.observeComponentCreation2((elmtId, isInitialRender) => {
            Column.create();
            Column.height('100%');
            Column.width('100%');
        }, Column);
        this.observeComponentCreation2((elmtId, isInitialRender) => {
            Image.create({ "id": 16777223, "type": 20000, params: [], "bundleName": "com.example.demo2", "moduleName": "entry" });
            Image.objectFit(ImageFit.Contain);
            Image.margin(15);
            Image.rotate({ x: 0, y: 0, z: 1, centerX: '50%', centerY: '50%',
                angle: this.rotateAngle });
        }, Image);
        Column.pop();
    }
    rerender() {
        this.updateDirtyElements();
    }
    static getEntryName(): string {
        return "Index";
    }
}
registerNamedRoute(() => new Index(undefined, {}), "", { bundleName: "com.example.demo2", moduleName: "entry", pagePath: "pages/Index", pageFullPath: "entry/src/main/ets/pages/Index", integratedHsp: "false" });
