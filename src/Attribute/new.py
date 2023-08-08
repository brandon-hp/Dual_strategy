package
pojo;

/ **
*水果实体类
* /
public


class Fruit {
/ **
* 编号
* /
private int Fid;

/ **
* 名称
* /
private String Fname;

/ **
* 单价
* /
private Double Fprice;

/ **
* 库存
* /
private int Fstock;

public Fruit(int fid, String fname, Double fprice, int fstock) {
Fid = fid;
Fname = fname;
Fprice = fprice;
Fstock = fstock;
}

public Fruit() {
this(0, "", 0.0, 0);
}

public void setFid(int fid) {
Fid = fid;
}

public void setFname(String fname) {
Fname = fname;
}

public void setFprice(Double fprice) {
if (fprice <= 0) {
throw new RuntimeException("定义"+Fid+"编号的水果的单价错误！！！");
}
Fprice = fprice;

}

public void setFstock(int fstock) {
if (fstock <= 0) {
throw new RuntimeException("定义"+Fid+"编号的水果的库存错误！！！");
}
Fstock = fstock;
}

public int getFid() {


return Fid;
}

public
String
getFname()
{
return Fname;
}

public
Double
getFprice()
{
return Fprice;
}

public
int
getFstock()
{
return Fstock;
}

@Override


public
boolean
equals(Object
o) {
if (this == o) return true;
if (o == null | | getClass() != o.getClass()) return false;
Fruit
fruit = (Fruit)
o;
return Fid == fruit.Fid & & Fstock == fruit.Fstock & & Fname.equals(fruit.Fname) & & Fprice == fruit.Fprice;
}

@Override


public
String
toString()
{
return "Fruit{" +
"Fid=" + Fid +
", Fname='" + Fname + '\'' +
", Fprice=" + Fprice +
", Fstock=" + Fstock +
'}';
}
}