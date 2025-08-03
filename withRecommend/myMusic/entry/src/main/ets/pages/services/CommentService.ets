import { getSongCommentList,addComment } from "../common/Api"
import {AddCommentParam} from '../model/paramModel'
import promptAction from '@ohos.promptAction'
class CommentService {
  //获取登录的用户信息
  async getSongCommentData() {
    const response = await getSongCommentList()
    if (response.success) {
      let data = JSON.stringify(response.data)
      console.log("获取歌曲评论列表：" + data)
      return JSON.parse(data)

    } else {
      promptAction.showToast({
        message: response.message,
        duration: 5000
      })
    }
  }
  async addCommentData(param:AddCommentParam){
    const response  =await addComment({ content: param.content, up: param.up, songId: param.songId })
    if (response.success) {
      this.getSongCommentData()
    }
  }
}

let commentService = new CommentService()
export default commentService