import router from '@ohos.router'
@Component
export struct NavBar {
  title?:string=""
  build(){
    Column(){
      Row(){
        Row() {
          Image($r("app.media.back")).fillColor(Color.Gray).height('36lpx')
        }.width("120lpx")
        .height('100%')
        .onClick(() => {
          router.back()
        })
        .justifyContent(FlexAlign.Center)
        Row(){
          if(this.title!=""){
            Text(this.title)
              .fontColor('#333333')
              .textAlign(TextAlign.Center)
              .fontSize('44lpx')
          }
        }
        .layoutWeight(1)
      }
      .height("40")
      .width('100%')
    }.backgroundColor(Color.White)
  }
}