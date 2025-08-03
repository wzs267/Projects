// 接口返回值
export class ResponseModel{
  success:boolean;
  message:string;
  data:Array<object>;
  constructor(){
    this.success = false;
    this.message = "";
    this.data = [];
  }
}

// 传值参数
export class ParamModel{
  cid?:number;
  id?:number;
  page?:number;
  keywords?:string;
  type?:number;
  action?:string;
  tabIndex?:number;
}

//登录
export  interface  LoginModel {
  username:string,
  password:string
}
// 添加评论传值
export interface AddCommentParam {
  content:string;
  up:number,
  songId:number
}
// 收藏
export interface isCollectParam {
  ids:string;
}
// 修改用户信息
export interface UserEditParam {
  username?:string;
  password?:string;
  avatar?:string;
  sex?:number;
  phone_num?:string;
  email?:string;
  birth?:string;
  introduction?:string;
  location?:string;

}
// 枚举类型 发布事件的ID
export enum EmitEventType {
  UPDATE_STATE
}