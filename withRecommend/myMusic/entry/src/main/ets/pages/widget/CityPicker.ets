import { proCityList } from "../services/CityServices"

@Component
export struct CityPicker {
  @State provincesList: string[] = []
  @State cityList: string[] = []
  @State countyList: string[] = []
  @State provincesIndex: number = 0
  @State cityIndex: number = 0
  @State countyIndex: number = 0
  onChange:(v:string)=>void=()=>{}

  aboutToAppear() {
    this.initData()
  }

  //刚开始初始化的数据
  initData() {
    //绑定数据 省
    for (let i = 0; i < proCityList.length; i++) {
      this.provincesList.push(proCityList[i].name)
    }
    //市
    for (let i = 0; i < proCityList[0].children.length; i++) {
      this.cityList.push(proCityList[0].children[i].name)
    }
    // 区
    for (let i = 0; i < proCityList[0].children[0].children.length; i++) {
      this.countyList.push(proCityList[0].children[0].children[i].name)
    }
  }

  build() {
    Row({ space: "20lpx" }) {
      TextPicker({ range: this.provincesList })
        .onChange((value: string | string[], index: number | number[]) => {
          console.info('省, value: ' + value + ', index: ' + index)
          //改变省的时候绑定省下面的市
          let tempList: string[] = []
          if (typeof index === 'number') {
            for (let i = 0; i < proCityList[index].children.length; i++) {
              tempList.push(proCityList[index].children[i].name)
            }
            this.cityList = tempList
            this.provincesIndex = index

            //改变省的时候绑定省下面第一个市的区
            let tempCountyList: string[] = []
            for (let i = 0; i < proCityList[index].children[0].children.length; i++) {
              tempCountyList.push(proCityList[index].children[0].children[i].name)
            }
            this.countyList = tempCountyList
            let proCityStr: string = proCityList[this.provincesIndex].name + "-" +
            proCityList[this.provincesIndex].children[this.cityIndex].name + "-" +
            proCityList[this.provincesIndex].children[this.cityIndex].children[this.countyIndex].name
            this.onChange(proCityStr)
          }

        }).layoutWeight(1)
      TextPicker({ range: this.cityList })
        .onChange((value: string | string[], index: number | number[]) => {
          if (typeof index === 'number') {
            console.info('市, value: ' + value + ', index: ' + index)
            let tempList: string[] = []
            for (let i = 0; i < proCityList[this.provincesIndex].children[index].children.length; i++) {
              tempList.push(proCityList[this.provincesIndex].children[index].children[i].name)
            }
            this.countyList = tempList
            this.cityIndex = index
            let proCityStr: string = proCityList[this.provincesIndex].name + "-" +
            proCityList[this.provincesIndex].children[this.cityIndex].name + "-" +
            proCityList[this.provincesIndex].children[this.cityIndex].children[this.countyIndex].name
            this.onChange(proCityStr)
          }

        }).layoutWeight(1)
      TextPicker({ range: this.countyList })
        .onChange((value: string | string[], index: number | number[]) => {
          if (typeof index === 'number') {
            console.info('Picker item changed, value: ' + value + ', index: ' + index)
            this.countyIndex = index
            let proCityStr: string = proCityList[this.provincesIndex].name + "-" +
            proCityList[this.provincesIndex].children[this.cityIndex].name + "-" +
            proCityList[this.provincesIndex].children[this.cityIndex].children[this.countyIndex].name
            this.onChange(proCityStr)
          }
        }).layoutWeight(1)
    }.width("100%")
  }
}