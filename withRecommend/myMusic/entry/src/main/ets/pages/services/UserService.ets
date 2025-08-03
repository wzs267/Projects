import preferencesUtil from "../common/PreferencesUtil"
import { UserModel } from '../model/AppModel'

class UserService {
  //获取登录的用户信息
  async getUserData():Promise<UserModel> {
    let userListData = await preferencesUtil.get("userinfoStore", "userinfo", "")
    if (userListData.toString() != "") {
      let tempList: UserModel = JSON.parse(userListData.toString())
      return tempList
    } else {
      return {}
    }
  }
  //退出登录
  async cleanUserData() {
    await preferencesUtil.set("userinfoStore", "userinfo", "")
  }
}

let userService = new UserService()
export default userService