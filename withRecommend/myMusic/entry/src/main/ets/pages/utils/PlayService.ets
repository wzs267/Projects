import preferences from '@ohos.data.preferences'
import { defaultState, PlayStateType } from '../model/PlayStateModel'
import preferencesUtil from '../common/PreferencesUtil'
/**
 * 首选项工具类，用于记录应用程序的状态
 * */
export class PlayService{
  StoreName = "MN_CLOUD_MUSIC"
  PlayStateKey = 'MN_PLAY_STATE'
  //存放数据
  async setState(data: PlayStateType) {
      await preferencesUtil.set(this.StoreName, this.PlayStateKey, JSON.stringify(data))
  }
  //获取历史记录
  async getState() {
    let playData = await preferencesUtil.get(this.StoreName, this.PlayStateKey, "")
    if (playData!='') {
      return JSON.parse(playData.toString())
    }else{
      return defaultState
    }

  }
}
let playService = new PlayService()

export default playService