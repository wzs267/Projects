import router from '@ohos.router'
import { UserModel } from './model/AppModel'
import inputMethod from '@ohos.inputMethod'
import preferencesUtil from './common/PreferencesUtil'
import promptAction from '@ohos.promptAction'
import { login } from './common/Api'
import { ConfigurationConstant } from '@kit.AbilityKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
@Entry
@Component
struct Index {
  @State darkMode: boolean = false;
  @State username: string = ''
  @State password: string = ''
  @StorageLink("isLogin") isLogin: boolean = false
  @StorageLink("userInfo") userInfo: UserModel = {}
  @State token: string = ''
  // aboutToAppear(): void {
  //   getContext(this)
  //     .getApplicationContext()
  //     .setColorMode(this.darkMode ?
  //     ConfigurationConstant.ColorMode.COLOR_MODE_DARK :
  //     ConfigurationConstant.ColorMode.COLOR_MODE_LIGHT);
  // }
  //
  // aboutToDisappear(): void {
  //   getContext(this)
  //     .getApplicationContext()
  //     .setColorMode(ConfigurationConstant.ColorMode.COLOR_MODE_NOT_SET);
  // }
  //登录
  async doLogin() {
    //隐藏键盘
    let im = inputMethod.getController()
    im.stopInputSession()
    const response = await login({
      username: this.username,
      password: this.password
    })
    console.log('response', JSON.stringify(response))
    if (response.success) {
      //执行登录
      //保存用户信息  preferencesUtil本地存储
      let user: string = JSON.stringify(response.data ? response.data["user"] : "")
      let token:string = response.data ? response.data["token"] : ""
      await preferencesUtil.set("userinfoStore", "userinfo", user)
      await preferencesUtil.set("userinfoStore", "token", token)
      //同步用户信息  状态管理
      this.isLogin = true
      this.userInfo = response.data ? response.data["user"] : ""
      PersistentStorage.persistProp("isLogin", this.isLogin)
      PersistentStorage.persistProp("userInfo", this.userInfo)
      router.back()
    } else {
      promptAction.showToast({
        message: response.message,
        duration: 5000
      })
    }

  }
  @Builder
  imageButton(src: Resource) {
    Button({ type: ButtonType.Circle, stateEffect: true }) {
      Image(src)
    }
    .height('48vp')
    .width('48vp')
    .backgroundColor('#F1F3F5')
  }
  build() {
    Column() {
      // Row() {
      //   Text('深色模式')
      //     .fontSize(16);
      //   Toggle({ type: ToggleType.Switch, isOn: this.darkMode })
      //     .onChange((isOn: boolean) => {
      //       hilog.info(0XFFFF, 'DarkMode', `Toggle.onChange2： isOn ${isOn}`);
      //       this.darkMode = isOn;
      //       getContext(this)
      //         .getApplicationContext()
      //         .setColorMode(this.darkMode ?
      //         ConfigurationConstant.ColorMode.COLOR_MODE_DARK :
      //         ConfigurationConstant.ColorMode.COLOR_MODE_LIGHT);
      //     });
      // }
      // .justifyContent(FlexAlign.SpaceBetween)
      // .width('90%');
      Image($r('app.media.logo'))
        .width('78vp')
        .height('78vp')
        .margin({ top: '150vp', bottom: '8vp' })
      Text('登录界面')
        .fontSize('24fp')
        .fontWeight(FontWeight.Medium)
        .fontColor($r('app.color.button_text_light'))
        .margin({
          bottom: '30vp',
          top: '8vp'
        })
      TextInput({ placeholder: '请输入用户' })
        .maxLength(11)
        .type(InputType.Normal)
        .placeholderColor($r('app.color.login_text'))
        .height('45vp')
        .fontSize('18fp')
        .backgroundColor('#F1F3F5')
        .width('328vp')
        .margin({ top: '24rpx' })
        .fontColor('#333333')
        .onChange((value: string) => {
          this.username = value;
        })
      TextInput({ placeholder: '请输入密码' })
        .maxLength(8)
        .type(InputType.Password)
        .placeholderColor($r('app.color.login_text'))
        .height('45vp')
        .fontSize('18fp')
        .backgroundColor('#F1F3F5')
        .width('328vp')
        .margin({ top: '24rpx' })
        .fontColor('#333333')
        .onChange((value: string) => {
          this.password = value;
        })
      Row() {
        Text('验证码登录')
          .fontColor($r('app.color.button_text'))
          .fontSize('14fp')
          .fontWeight(FontWeight.Medium)
        Text('忘记密码')
          .fontColor($r('app.color.button_text'))
          .fontSize('14fp')
          .fontWeight(FontWeight.Medium)
      }
      .justifyContent(FlexAlign.SpaceBetween)
      .width('328vp')
      .margin({ top: '8vp' })

      Button('登录', { type: ButtonType.Capsule })
        .width('328vp')
        .height('40vp')
        .fontSize('16fp')
        .fontWeight(FontWeight.Medium)
        .backgroundColor($r('app.color.button_background'))
        .margin({
          top: '48vp',
          bottom: '12vp'
        })
        .onClick(() => {
          this.doLogin()
        })
      Text('注册账号')
        .fontColor($r('app.color.button_text'))
        .fontSize('16vp')
        .fontWeight(FontWeight.Medium)
      Blank()
      Text('其他登录方式')
        .fontColor('#838D97')
        .fontSize('12fp')
        .fontWeight(FontWeight.Medium)
        .margin({
          top: '50vp',
          bottom: '12vp'
        })
      Row({ space:'80rpx' }) {
        this.imageButton($r('app.media.login_method1'))
        this.imageButton($r('app.media.login_method2'))
        this.imageButton($r('app.media.login_method3'))
      }
      .margin({ bottom: '16vp' })
    }
    .height('100%')
    .width('100%')
    .backgroundColor($r('app.color.page_background'))
    .padding({
      left: '12vp',
      right: '12vp',
      bottom: '24vp'
    })
  }
}