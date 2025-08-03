import { SongModel} from '../model/AppModel'
import { isCollect,setCollect  } from '../common/Api'
import { isCollectParam,ParamModel } from '../model/paramModel'
import promptAction from '@ohos.promptAction'

class CollectService {
  //获取登录的用户信息
  // 判断是否收藏
  async isCollectData(SongList:SongModel[]){
    const params :isCollectParam = {
      // 获得歌曲的所有id
      ids:SongList.map(item => item.id).join(',')
    }
    const response = await isCollect(params)
    if (response.success) {
      let data = JSON.stringify(response.data)
      return  JSON.parse(data)   // 返回接口结果
    } else {
      promptAction.showToast({
        message: response.message,
        duration: 5000
      })
    }
  }
  // 添加取消收藏
  async setCollectData(id:number,action:string):Promise<boolean>{
    const params :ParamModel = {
      // 获得歌曲的所有id
      id:id,
      action:action
    }
    const response = await setCollect(params)
    if (response.success) {
      console.log("收藏成功")
      return true
    }else{
      promptAction.showToast({
        message: response.message,
        duration: 5000
      })
      return false
    }
  }
}

let collectService = new CollectService()
export default collectService