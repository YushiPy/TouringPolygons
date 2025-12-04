
from math import pi, tau

from problem1_draw import Drawing
from vector2 import Vector2
from polygon2 import Polygon2

def regular(n: int, r: float, start: Vector2 = Vector2(), angle: float = 0) -> Polygon2:
	"""
	Create a regular polygon with `n` vertices and radius `r`.

	:param int n: The number of vertices.
	:param float r: The radius of the polygon.

	:return: A Polygon2 object representing the regular polygon.
	"""
	return Polygon2(start + Vector2.from_polar(r, i * tau / n + angle) for i in range(n))

sol = Drawing(Vector2(-3, 0), Vector2(3, 0), [
	regular(5, 1, start=Vector2(-1, 3), angle=0.1),
	Polygon2([Vector2(-1, -1), Vector2(1, -1), Vector2(1, 1), Vector2(-1, 1)]),
])

test1 = Drawing(
	Vector2(5, 1), 
	Vector2(7, 3),
	[
		Polygon2([Vector2(3, 0), Vector2(2, 4), Vector2(1, 4), Vector2(-1, 1)]),
		Polygon2([Vector2(3, 3), Vector2(4, 3), Vector2(4, 4), Vector2(3, 4)]),
		Polygon2([Vector2(5, 5), Vector2(6, 5), Vector2(6, 6), Vector2(5, 6)]),
	]
)

test2 = Drawing(
	Vector2(-1, -1),
	Vector2(1, -1),
	[
		Polygon2([Vector2.from_polar(2, i * tau / 6 + pi * 0.35) + Vector2(4, 5) for i in range(6)]),
		Polygon2([Vector2.from_polar(2, i * tau / 3 + pi /4) + Vector2(-3, 4) for i in range(3)]),
		Polygon2([Vector2.from_polar(2, i * tau / 10) + Vector2(5, -4) for i in range(10)]),
		Polygon2([Vector2.from_polar(2, i * tau / 4 + pi / 4) + Vector2(-4, -2) for i in range(4)]),
		Polygon2([Vector2.from_polar(2, i * tau / 30 + pi / 4) + Vector2(0, -8) for i in range(30)]),
	]
)

test3 = Drawing(
	Vector2(4, 1), 
	Vector2(7, 3),
	[
		Polygon2([Vector2(3, 0), Vector2(1, 4), Vector2(-1, 1)]),
		Polygon2([Vector2(2.5, 5.), Vector2(4.7, 5), Vector2(4, 6), Vector2(3, 6)]),
		Polygon2([Vector2(5, 5), Vector2(6, 5), Vector2(6, 6), Vector2(5, 6)])
	]
)

test4 = Drawing(
	Vector2(4, 1), 
	Vector2(7, 3),
	[
		Polygon2([Vector2(-1, 1), Vector2(1, 4), Vector2(3, 0), Vector2(-1, 0)]),
	]
)

test5 = Drawing(
	Vector2(-1, -1),
	Vector2(1, -1),
	[
		Polygon2([Vector2.from_polar(2, i * tau / 6 + pi * 0.35) + Vector2(4, 5) for i in range(6)]),
		Polygon2([Vector2.from_polar(2, i * tau / 3) + Vector2(5, -4) for i in range(3)]),
		Polygon2([Vector2.from_polar(2, i * tau / 4 + pi / 4) + Vector2(-4, -2) for i in range(4)]),
	]
)

test6 = Drawing(
	Vector2(-3, 0),
	Vector2(0, 2),
	[
		Polygon2(regular(3, 1, Vector2(0, 0), pi / 3))
	]
)

test7 = Drawing(
	Vector2(5, 1), 
	Vector2(7, 3),
	[
		Polygon2([Vector2(3, 0), Vector2(2, 4), Vector2(1, 2)]),
		Polygon2([Vector2(3, 3), Vector2(5, 3), Vector2(4.5, 4), Vector2(3.5, 4)]),
		regular(5, 1.3, Vector2(5, 6), 0.1),
	]
)






test8 = Drawing(Vector2(-1.8311630107011039, -2.0455863927678903), Vector2(0.7037859997353402, 1.9802048781724153), [(Vector2(-0.008915251813969016, 1.4484784591628075), Vector2(-0.8574691690535456, 0.0936205330361633), Vector2(0.497388757073098, -0.7549333842034134), Vector2(1.3459426743126752, 0.5999245419232302)), (Vector2(-0.9184897785493833, -3.745349669328246), Vector2(-1.461341682860255, -4.157995879074984), Vector2(-1.3754055345996539, -4.834442523574327), Vector2(-0.7466174820281806, -5.098242958326932), Vector2(-0.20376557771730885, -4.685596748580195), Vector2(-0.28970172597791033, -4.009150104080851)), (Vector2(-2.626169682147251, 2.4303775677941433), Vector2(-2.861647716995209, 1.478874177939144), Vector2(-2.029480932366719, 0.9608905407820508), Vector2(-1.2796955403096413, 1.5922624372576735), Vector2(-1.6484694683786913, 2.500455365978181)), (Vector2(-3.4635859336631096, -4.0210169450448445), Vector2(-4.140685766548763, -1.9624561179959676), Vector2(-5.584901821565768, -3.57812218769758)), (Vector2(-0.9400834681022241, -1.7599114512126168), Vector2(-0.8280792309802324, -2.5095401484297586), Vector2(-0.17216241572384763, -2.8893575575735073), Vector2(0.5337482375141906, -2.6133534226954325), Vector2(0.7580876088925131, -1.8893644868170398), Vector2(0.3319235760740411, -1.2625691853376457), Vector2(-0.4238336520668973, -1.2049571645652932))])
test9 = Drawing(Vector2(-0.6689245721488486, 1.384636454242803), Vector2(-1.7944024140206347, 0.8561068373774647), [(Vector2(0.3891530771855134, 0.42018482516746847), Vector2(-0.4317981831553246, -0.23746230279628833), Vector2(0.22584894480843215, -1.0584135631371265), Vector2(1.0468002051492702, -0.40076643517336985)), (Vector2(2.3308175591223876, 2.6044217209232845), Vector2(2.5674627183612526, 2.953409087433172), Vector2(2.652542654181857, 3.3663915542467837), Vector2(2.5731047176070527, 3.780496284487131), Vector2(2.341242614373665, 4.13267958684385), Vector2(1.9922552478637772, 4.369324746082716), Vector2(1.5792727810501657, 4.45440468190332), Vector2(1.165168050809818, 4.374966745328516), Vector2(0.8129847484530985, 4.143104642095128), Vector2(0.5763395892142333, 3.7941172755852404), Vector2(0.49125965339362887, 3.381134808771628), Vector2(0.5706975899684332, 2.9670300785312813), Vector2(0.802559693201821, 2.614846776174562), Vector2(1.1515470597117088, 2.3782016169356965), Vector2(1.5645295265253192, 2.293121681115092), Vector2(1.9786342567656678, 2.3725596176898964)), (Vector2(-1.463107054678766, -3.3138108716939243), Vector2(-1.2181189079830754, -3.2314336720517245), Vector2(-1.0278169250414013, -3.0565326991249075), Vector2(-0.9251060475436006, -2.8193499200579786), Vector2(-0.9277459193080707, -2.5608963920895116), Vector2(-1.035280082524253, -2.3258610781818447), Vector2(-1.2291149034548663, -2.154883719338408), Vector2(-1.475734580626941, -2.0775278541837734), Vector2(-1.7324963313879935, -2.107169014559151), Vector2(-1.9550037164241634, -2.238681974554149), Vector2(-2.1047831874465612, -2.449326948627898), Vector2(-2.155936513211639, -2.7026815071135353), Vector2(-2.099618818981239, -2.954938345194986), Vector2(-1.9455679457179862, -3.1624799638865384), Vector2(-1.7204206889173461, -3.2894205318600074)), (Vector2(8.602221952759983, -6.566534009321133), Vector2(8.446706750823765, -7.734233841960569), Vector2(9.262689099612693, -8.583869460119518), Vector2(10.435717647482175, -8.47564791391617), Vector2(11.082477972464812, -7.49106223495888), Vector2(10.715946358342356, -6.371525521390175), Vector2(9.612128586431378, -5.960071752914403)), (Vector2(4.491534431176148, 1.8385500245290738), Vector2(4.878035828565324, 2.297645352010318), Vector2(4.879012288123093, 2.8977710858575074), Vector2(4.494006913570238, 3.3581217255489406), Vector2(3.9031679985181573, 3.4632940905455287), Vector2(3.382955637762827, 3.1640768624047304), Vector2(3.176782976418057, 2.6004771076026993), Vector2(3.3811204940809616, 2.0362094151248673), Vector2(3.900356395278522, 1.7353009094184748))])

test10 = Drawing(
	Vector2(-10, 0),
	Vector2(10, 0),
	[
		regular(5, 2, Vector2(-5, 5), 0.1),
		regular(4, 2, Vector2(2, 4), 0.3),
		regular(6, 3, Vector2(7, -6), 2),
		regular(3, 1.5, Vector2(-5, -4), 0.5),
	]
)

#test1.draw([0], False)
#test2.draw([0], False)
#test3.draw([0], False)
#test4.draw([0], False)
#test5.draw([0], False)
#test6.draw([0], False)
#test7.draw([0], False)
#test8.draw([0], False)
#test9.draw([0], False)

test10.draw([0], False)