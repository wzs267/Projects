//getCateMenu
exports.getCateMenu=(rootList, pid, list) =>{
  //初始值 pid 0,第二次调用 pid  1
  for (let i = 0; i < rootList.length; i++) {
    let item = rootList[i];
    if (item.parentId == pid) {
      list.push(item);
    }
  }
  console.log(list);
  list.map((item) => {
    item.children = [];
    this.getCateMenu(rootList, item.id, item.children);
    if (item.children.length == 0) {
      delete item.children;
    }
  });
  return list;
}

// getCommentMenu

exports.getCommentMenu=(rootList, pid, list) =>{
  //初始值 pid 0,第二次调用 pid  1
  for (let i = 0; i < rootList.length; i++) {
    let item = rootList[i];
    if (item.replyId == pid) {
      list.push(item);
    }
  }
  console.log(list);
  list.map((item) => {
    item.children = [];
    this.getCommentMenu(rootList, item.id, item.children);
    if (item.children.length == 0) {
      delete item.children;
    }
  });
  return list;
}

function date(format, timeStamp) {
  if ("" + timeStamp.length <= 10) {
    timeStamp = +timeStamp * 1000;
  } else {
    timeStamp = +timeStamp;
  }
  let _date = new Date(timeStamp),
    Y = _date.getFullYear(),
    m = _date.getMonth() + 1,
    d = _date.getDate(),
    H = _date.getHours(),
    i = _date.getMinutes(),
    s = _date.getSeconds();

  m = m < 10 ? "0" + m : m;
  d = d < 10 ? "0" + d : d;
  H = H < 10 ? "0" + H : H;
  i = i < 10 ? "0" + i : i;
  s = s < 10 ? "0" + s : s;

  return format.replace(/[YmdHis]/g, (key) => {
    return { Y, m, d, H, i, s }[key];
  });
}
exports.createOrderNumber= ()=>{
  
    let random_no = date("Ymd", +new Date());
    for (let i = 0; i < 12; i++) {
      random_no += Math.floor(Math.random() * 10);
    }
    return random_no;
  
}

exports.orderStatus = [
        { status: 0, name: "待付款" },
        { status: 1, name: "待发货" },
        { status: 2, name: "已发货" },
        { status: 3, name: "已完成" },
        { status: 4, name: "已取消" },
];