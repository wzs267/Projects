import preferencesUtil from "../common/PreferencesUtil"
class SearchService {
  async setHistoryData(keywords: string) {
    let searchListData = await preferencesUtil.get("searchStore", "searchList", "")
    if (searchListData.toString() != "") {
      //判断本地存储中有没有当前数据，如果有不做操作、 如果没有当前数据,本地存储的数据和当前数据拼接后重新写入
      let tempList: string[] = JSON.parse(searchListData.toString())
      const index = tempList.findIndex(item=>item==keywords)
      if (index > -1) { //存在删除
        tempList.splice(index,1)
      }
      tempList.unshift(keywords)
      if(tempList.length > 10){
        tempList.length = 10;
      }
      //保存搜索关键词
      await preferencesUtil.set("searchStore", "searchList", JSON.stringify(tempList))
    } else {
      let tempList: string[] = []
      tempList.push(keywords)
      //保存搜索关键词
      await preferencesUtil.set("searchStore", "searchList", JSON.stringify(tempList))
    }

  }

  //获取历史记录
  async getHistoryData() {
    let searchListData = await preferencesUtil.get("searchStore", "searchList", "")
    if (searchListData.toString() != "") {
      let tempList: string[] = JSON.parse(searchListData.toString())
      return tempList
    } else {
      return []
    }
  }

  //删除历史记录
  async deleteHistoryData(keywords: string) {
    // ["手机","电脑",'女']   电脑     ["手机",'女']
    let searchListData = await preferencesUtil.get("searchStore", "searchList", "")
    if (searchListData.toString() != "") {
      let tempList: string[] = JSON.parse(searchListData.toString())
      tempList = tempList.filter((item) => {
        return item != keywords
      })
      await preferencesUtil.set("searchStore", "searchList", JSON.stringify(tempList))
    }
  }

  //清空历史记录
  async cleanHistoryData() {
    await preferencesUtil.set("searchStore", "searchList", "")
  }
}

let searchService = new SearchService()

export default searchService