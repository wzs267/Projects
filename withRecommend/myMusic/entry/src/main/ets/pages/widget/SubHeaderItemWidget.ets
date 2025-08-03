import { searchCateNameModel } from '../model/AppModel'

@Component
export struct SubHeaderItemWidget {
  @ObjectLink item:searchCateNameModel
  @Prop index:number
  @Link selectedSubHeaderIndex:number
  func:(key: number)=>void = ()=>{}  //接收父组件的方法
  build(){
    Row() {
      Text(this.item.title)
        .fontSize("38lpx")
        .textAlign(TextAlign.Center)
        .fontColor(this.selectedSubHeaderIndex == this.index  ? Color.Red : Color.Black)
    }.layoutWeight(1)
    .justifyContent(FlexAlign.Center)
    .onClick(() => {
      this.func(this.index)
    })
  }

}