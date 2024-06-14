import sympy as sp
from sympy import symbols, Function, Eq, dsolve, pretty, integrate, simplify, solve, expand
import re
t = symbols('t')
y = Function('y')
A,B,C,D,E,F,G,H = symbols('A B C D E F G H')


def get_part(a,b,c,p_x,comp="0"):
    print("The particular solution is: ")
    p_x = f"{p_x}"
    part = undet_coef(p_x)
    part_copy = p_x
    if "sin" in part_copy and "sin" in comp:
        if get_sin_cos_coe(part_copy) == get_sin_cos_coe(comp):
            print(f"sin({get_sin_cos_coe(part_copy)}*t) is in both equations, therefore multiply through by t")
            part = sp.simplify(sp.sympify(f"({part}) * t"))
            print(f"The updated particular solution is, {part}")
    if "exp" in part_copy:
        if get_exp_coe(part_copy) == get_exp_coe(comp):
            print(f"exp({get_exp_coe(part_copy)}*t) is in both equations, therefore multiply through by t")
            part = sp.simplify(sp.sympify(f"({part}) * t"))
            print(f"The updated particular solution is, {part}")
            if f"t*exp({get_exp_coe(comp)}*t)" in comp:
                print(f"t*exp({get_exp_coe(part_copy)}*t) is in both equations, therefore multiply through by t")
                part = sp.simplify(sp.sympify(f"({part}) * t"))
                print(f"The updated particular solution is, {part}")
    part = sp.sympify(part)
    y_1st_deriv = sp.simplify(part.diff(t,1))
    y_2nd_deriv = sp.simplify(part.diff(t,2))
    print(f"y = {part}")
    print(f"y` = {y_1st_deriv}")
    print(f"y`` = {y_2nd_deriv}")
    print("Substituting")
    print(f"{a}*({y_2nd_deriv}) + {b}*({y_1st_deriv}) + {c}*({part}) = {p_x}")
    eqn_1 = sp.simplify(sp.sympify(f"{a}*({y_2nd_deriv}) + {b}*({y_1st_deriv}) + {c}*({part})"))
    print(f"{eqn_1} = {p_x}")
    p_x = sp.sympify(p_x)
    eqn_2 = Eq(eqn_1, p_x)
    print("Comparing coefficients")
    if "t*cos" in part_copy or "t*sin" in part_copy:
        solutions = solve(eqn_2,(A,B,C,D))
        print(solutions)
        a,b,c,d = solutions[A], solutions[B], solutions[C], solutions[D]
        print(f"A = {a}, B = {b}, C = {c}, D = {d}")
        print(f"y = {part.subs(A,a).subs(B,b).subs(C,c).subs(D,d)}")
        return part.subs(A,a).subs(B,b).subs(C,c).subs(D,d)
    elif "t*exp" in part_copy:
        solutions = solve(eqn_2,(A,B))
        print(f"A = {solutions[A]}, B = {solutions[B]}")
        a,b = solutions[A],solutions[B]
        print(part.subs(A,a).subs(B,b))
        return part.subs(A,a).subs(B,b)
    elif "exp" in part_copy:
        solutions = solve(eqn_2,(A))
        print(f"A = {solutions[0]}")
        a = solutions[0]
        print(part.subs(A,a))
        return part.subs(A,a)
    elif "sin" in part_copy or "cos" in part_copy:
        solutions = solve(eqn_2,(A,B))
        print(solutions)
        a,b = solutions[A], solutions[B]
        print(f"A = {a}, B = {b}")
        print(f"y = {part.subs(A,a).subs(B,b)}")
        return part.subs(A,a).subs(B,b)
    
    else:
        degree = get_degree(part_copy)
        if degree == 0:
            solution = solve(eqn_2,A)
            a = solution[0]
            print(f"A = {a}")
            print(f"y = {part.subs(A,a)}")
            return part.subs(A,a)
        elif degree == 1:
            solution = solve(eqn_2,(A,B))
            a = solution[A]
            b = solution[B]
            print(f"A = {a}, B = {b}")
            print(f"y = {part.subs(A,a).subs(B,b)}")
            return part.subs(A,a).subs(B,b)
        elif degree == 2:
            solution = solve(eqn_2,(A,B,C))
            a = solution[A]
            b = solution[B]
            c = solution[C]
            print(f"A = {a}, B = {b}, C = {c}")
            print(f"y = {part.subs(A,a).subs(B,b).subs(C,c)}")
            return part.subs(A,a).subs(B,b).subs(C,c)
        elif degree == 3:
            solution = solve(eqn_2,(A,B,C,D))
            a = solution[A]
            b = solution[B]
            c = solution[C]
            d = solution[D]
            print(f"A = {a}, B = {b}, C = {c}, D = {d}")
            print(f"y = {part.subs(A,a).subs(B,b).subs(C,c).subs(D,d)}")
            return part.subs(A,a).subs(B,b).subs(C,c).subs(D,d)
    
def get_part_special(a,b,c,p_x,comp):
    p_x_comps = split_equation(p_x)
    solution = sp.sympify(0)
    sol_array = []
    print("Given the nature of p(t), it was broken into the components and the particular solution for each will be found")
    for val in p_x_comps:
        if val=="":
            continue
        print(f"For {val},")
        sol = get_part(a,b,c,val,comp)
        solution+=sol
        sol_array.append(sol)
    print("y_part =", end="")
    for val in sol_array:
        if val == sol_array[0]:
            print(f"{val} ",end="")
        else:
            print(f"+ {val}",end="")
    print("\n")
    return solution
    
def split_equation(equation):
    equation = equation.replace(" ", "")
    equation = equation.replace("+", " +").replace("-"," -").replace("( ","(")
    comps = equation.split(" ")
    return comps

def get_degree(txt): #This returns the degree of the passed polynomial
    x = txt.replace("**","^")
    num_of_t = len(re.findall("t",x))
    if num_of_t == 1:
        if not ("^" in x):
            return 1
    elif num_of_t==0:
        return 0
    power = re.findall(r"t.(\d+)",x)
    for num in range(len(power)):
        power[num] = int(power[num])
    power.sort(reverse=True)
    return power[0]

def get_sin_cos_coe(txt):
    value=1
    if "cos" in txt:
        if "-" in txt:
            value *= -1
            txt = txt.replace("-","")
        power = re.findall(r"cos\((\d+)",txt)
        if power==[]:
            value*=1
            return value
        return value*int(power[0])
    if "sin" in txt:
        if "-" in txt:
            value *= -1
            txt = txt.replace("-","")
        power = re.findall(r"sin\((\d+)",txt)
        if power==[]:
            value*=1
            return value
        return value*int(power[0])

def get_exp_coe(expression, variable="t"):
    start = expression.find("exp")
    expression=expression[start:]
    pattern = re.compile(r'([-+]?\d*\.\d+|[-+]?\d+)[*]' + re.escape(variable))
    match = pattern.search(expression)
    if match:
        return int(match.group(1))
    if "exp(t)" in expression:
        return 1
    if "exp(-t)" in expression:
        return -1
    else:
        # If no match is found, return 0
        return 0.0
    
def get_exp_coef(txt): #This returns the coefficient of t
    power = re.findall(r"exp.(\d+)",txt)
    if power==[]:
        return 1
    return int(power[0])

def undet_coef(eqn): #This returns the format of the passed equation in undetermined coefficients
    A,B,C,D,E,F = symbols('A B C D E F')
    eqn_copy = eqn.replace("**","^")
    if "t*" in eqn_copy:
        if "cos" in eqn_copy or "sin" in eqn_copy:
            coe = get_sin_cos_coe(eqn)
            if coe == 1:
                sol = f"(A*t + B)*sin(t) + (C*t + D)*cos(t)"
            else:
                sol = f"(A*t + B)*sin({coe}*t) + (C*t + D)*cos({coe}*t)"
            return sol
        elif "exp" in eqn:
            coe = get_exp_coe(eqn)
            if coe==1:
                sol = f"(A*t + B)*exp(t)"
            else:
                sol = f"(A*t + B)*exp({coe}*t)"
            return sol
    if "cos" in eqn or "sin" in eqn:
        if "exp" in eqn:
            coe_sin = get_sin_cos_coe(eqn)
            print(coe_sin)
            coe_exp = get_exp_coe(eqn)
            print(coe_exp)
            sol = f"({A}*cos({coe_sin}*t) + {B}*sin({coe_sin}*t))*exp({coe_exp}*t)"
            return sol
        coe = get_sin_cos_coe(eqn)
        if coe == 1:
            sol = f"{A}*cos(t) + {B}*sin(t)"
        else:
            sol = f"{A}*cos({coe}*t) + {B}*sin({coe}*t)"
        return sol
    elif "exp" in eqn:
        coe = get_exp_coe(eqn)
        if coe == 1:
            sol = f"{A}*exp(t)"
        else:
            sol = f"{A}*exp({coe}*t)"
        return sol
    else:
        degree = get_degree(eqn)
        if degree == 0:
            sol = f"{A}"
            return sol
        if degree == 1:
            sol = f"{A}*t + {B}"
            return sol
        if degree == 2:
            sol = f"{A}*t**(2) + {B}*t + {C}"
            return sol
        if degree == 3:
            sol = f"{A}*t**(3) + {B}*t**(2) + {C}*t + {D}"
            return sol

def first_order():
    order=1
    print("a*y' + b*y = p(x)")
    a=sp.sympify(input("Enter the value of a: "))
    b=sp.sympify(input("Enter the value of b: "))
    p_x=input("Enter the value of p(x): ")
    if p_x =="0":
        p_x=int(p_x)
    if p_x!=0:
        p_x=sp.sympify(p_x)
    equation=Eq(a*y(t).diff(t) + b*y(t), p_x)
    solve =0
    q=0
    try:
        ics=input("Are there initial conditions?(y/n) ")
        if ics.lower()=='y':
            print("y(t)=q")
            solve=int(input("Enter the value of t: "))
            q=int(input("Enter the value of q: "))
            initial_condition={y(solve):q}
            solution = expand(dsolve(equation, y(t),ics=initial_condition).rhs)
        elif ics.lower()=='n':
            solution = expand(dsolve(equation, y(t)).rhs)
        print(f"The solution to the differential equation ({a})*y' + ({b})*y = {p_x} is: ")
        print(f"y({t})={solution}")
    except:
        solution = "Can't solve"
        print(solution)
    return a,b, p_x, ics, solve, q, solution

def first_order_solver(a,b,p_x): #This is the steps function for first order
    print(f"({a})*y' + ({b})*y = {p_x}")
    print("y' + (b)*y = (p(t))")
    print("Dividing through by the coefficient of y` ")
    print(f"y` + {sp.sympify(b)/sp.sympify(a)}*y = {sp.sympify(p_x)/sp.sympify(a)}")
    print("Comparing Coefficients...")
    print(f"a={1}, b={sp.sympify(b)/sp.sympify(a)}, p(t)={sp.sympify(p_x)/sp.sympify(a)}")
    b = sp.sympify(b)/sp.sympify(a)
    p_x = sp.sympify(p_x)/sp.sympify(a)
    print("q(t)=∫(b)dt")
    text=f"q(t)=∫({b})dt"
    print(text)
    q_x=integrate(b,t)
    print(f"∫({b})dx = {q_x}")
    meu=sp.sympify(f"exp({q_x})")
    mu=symbols('mu')
    exp=sp.sympify(f"exp({q_x})")
    mu=pretty(mu, use_unicode=True)
    eqn1=f"{mu}(t)={exp}"
    print(f"{mu}(t) = exp(q(t))")
    print(f"{mu}(t) = exp({q_x})")
    print(f"exp({q_x}) = {sp.sympify(f'exp({q_x})')}")
    print(f"y(t)= (1/{mu}(t))*∫({mu}(t) * p(t))dx")
    print(f"y(t) = (1/{exp}) * ∫({exp} * {p_x})dx")
    eqn2=sp.sympify(f"{exp} * {p_x}")
    C1=symbols("C1")
    eqn3=integrate(eqn2,t) + C1
    print(f"∫({exp} * {p_x})dx = {eqn3}")
    print(f"(1/{exp}) * ({eqn3}) = {sp.simplify(sp.sympify(f'1/{exp}') * (eqn3))}")
    eqn4 = sp.simplify(sp.sympify(f'1/{exp}') * eqn3)
    print(f"y(t) = {expand(eqn4)}")
    return eqn4
    
def first_order_solver_ivp(a,b,p_x, ivp_in, ivp_out): #This is the steps function for first order
    print(f"({a})*y' + ({b})*y = {p_x}")
    print("y' + (b)*y = (p(t))")
    print("Dividing through by the coefficient of y` ")
    print(f"y` + {sp.sympify(b)/sp.sympify(a)}*y = {sp.sympify(p_x)/sp.sympify(a)}")
    print("Comparing Coefficients...")
    print(f"a={1}, b={sp.sympify(b)/sp.sympify(a)}, p(t)={sp.sympify(p_x)/sp.sympify(a)}")
    b = sp.sympify(b)/sp.sympify(a)
    p_x = sp.sympify(p_x)/sp.sympify(a)
    print("q(t)=∫(b)dt")
    text=f"q(t)=∫({b})dt"
    print(text)
    q_x=integrate(b,t)
    print(f"∫({b})dx = {q_x}")
    meu=sp.sympify(f"exp({q_x})")
    mu=symbols('mu')
    exp=sp.sympify(f"exp({q_x})")
    mu=pretty(mu, use_unicode=True)
    eqn1=f"{mu}(t)={exp}"
    print(f"{mu}(t) = exp(q(t))")
    print(f"{mu}(t) = exp({q_x})")
    print(f"y(t)= (1/{mu}(t))*∫({mu}(t) * p(t))dt")
    print(f"y(t) = (1/{exp}) * ∫({exp} * {p_x})dt")
    eqn2=sp.sympify(f"{exp} * {p_x}")
    C1=symbols("C1")
    eqn3=integrate(eqn2,t) + C1
    print(f"∫({exp} * {p_x})dt = {eqn3}")
    print(f"(1/{exp}) * ({eqn3}) = {sp.simplify(sp.sympify(f'1/{exp}') * (eqn3))}")
    eqn4 = sp.simplify(sp.sympify(f'1/{exp}') * eqn3)
    print(f"y(t) = {(eqn4)}")#Everything after this is for IVPs
    print(f"Given the condition y({ivp_in}) = {ivp_out}, then")
    print(f"y({ivp_in}) = {eqn4.subs(t, ivp_in)} = {ivp_out}")
    sol = solve(eqn4, C1)
    print(f"{C1} = {ivp_out} + {sol[0].subs(t,ivp_in)}")
    value=ivp_out + sol[0].subs(t,ivp_in)
    print(f"{C1} = {value}")
    print("Therefore, ")
    print(f"y(t) = {expand(eqn4.subs(C1, value))}")
    return eqn4.subs(C1, value)

#This is the second order equation solver
def second_order():
    order=2
    t = symbols('t')
    y = Function('y')
    print(f'a*{y}`` +b*{y}` +c*{y} = p({t})')
    steps = False
    a=input("Enter the value of a: ")
    b=input("Enter the value of b: ")
    c=input("Enter the value of c: ")
    try:
        a,b,c = int(a), int(b), int(c)
        steps=True
    #if (a.isdigit() and b.isdigit() and c.isdigit()):
    except:
        a,b,c = a,b,c
        steps = False    
    a=sp.sympify(a)
    b=sp.sympify(b)
    c=sp.sympify(c)
    p_x=input("Enter the value of p(x): ") #Note the chnage here
    p_x_copy = p_x
    if p_x!=0:
        p_x_without_t = p_x.replace("*t","").replace("t","").replace(" ","").replace("+","").replace("-","").replace("**","")
        if p_x_without_t.isdigit() or p_x_without_t.isdigit()=="":
            steps = steps and True
        elif not("cos" in p_x) and not("sin" in p_x):
            if not("exp" in p_x) and not("t +" in p_x) and not ("+ t" in p_x) and not ('t+' in p_x) and not ('+t' in p_x) and not ("- t" in p_x) and not ('t-' in p_x) and not ('-t' in p_x) : #This ensures there is no combination
                steps = steps and False
    p_x=sp.sympify(p_x)
    equation=Eq(a*y(t).diff(t,2) + b*y(t).diff(t) + c*y(t), p_x)
    try:
        typee = "none"
        solve1, solve2 = 0,0
        q1,q2=0,0
        ics=input("Are there initial conditions?(y/n) ")
        if ics.lower()=='y':
            typee= input("IVP or BVP? ")
            if typee.upper()=="IVP":
                print("y(t1)=q1, y`(t2)=q2")
                solve1=float(input("Enter the value of t1: "))
                solve2=float(input("Enter the value of t2: "))
                q1=float(input("Enter the value of q1: "))
                q2=float(input("Enter the value of q2: "))
                initial_condition={y(solve1):q1,y(t).diff(t).subs(t,solve2):q2}
                solution = dsolve(equation, y(t),ics=initial_condition).rhs
                print(f'The solution to the differential equation ({a})*y`` + ({b})*y` + ({c})*y = {p_x}, y({solve1})={q1}, y`({solve2})={q2} is: ')
            elif typee.upper()=="BVP":
                print("y(t1)=q1, y(t2)=q2")
                solve1=sp.sympify(input("Enter the value of t1: "))
                solve2=sp.sympify(input("Enter the value of t2: "))
                q1=int(input("Enter the value of q1: "))
                q2=int(input("Enter the value of q2: "))
                initial_condition={y(solve1):q1,y(solve2):q2}
                solution = dsolve(equation, y(t),ics=initial_condition).rhs
                print(f'The solution to the differential equation ({a})*y`` + ({b})*y` + ({c})*y = {p_x}, y({solve1})={q1}, y({solve2})={q2} is: ')
        elif ics.lower()=='n':
            solution = dsolve(equation, y(t)).rhs
            print(f'The solution to the differential equation ({a})*y`` + ({b})*y` + ({c})*y = {p_x} is: ')
        print(f"y(t)={expand(sp.simplify(solution))}")
    except:
        solution = "Can't solve, please check your input"
        print(solution)
    return a,b,c,p_x_copy, steps, ics, typee, solve1, solve2, q1, q2,solution

#Second Order solver with steps
def second_order_solver(a,b,c,p_x):
    p_x_copy = p_x
    a,b,c,p_x = sp.sympify(a), sp.sympify(b), sp.sympify(c), sp.sympify(p_x)
    print(f"({a})*y`` + ({b})*y` + ({c})*y = {p_x}")
    print("a*y`` + b*y` +c*y = (p(t))")
    print("Comparing Coefficients...")
    print(f"a = {a}, b = {b}, c = {c}, p(t) = {p_x}")
    print(f"Therefore the complementary equation is: \n{a}*r**(2) + {b}*r + {c} = 0")
    print("d = b**(2) - (4*a*c)")
    print(f"d = {b}**2 - (4*{a}*{c})")
    print(f"d = {b**2} - {4*a*c}")
    d = b**(2) - (4*a*c)
    print(f"d = {d}")
    if d!=0:
        if d>0:
            print("r = (-b \u00B1 sqrt(d))/(2*a) ")
            print((f"r = ({-b} \u00B1 sqrt({d}))/(2*{a})"))
            print(f"r = ({-b} \u00B1 {sp.sympify(f'sqrt({d})')})/{sp.sympify(2*a)}")
            eqn1 = sp.sympify(f"({-b} + sqrt({d}))/{2*a}")
            eqn2 = sp.sympify(f"({-b} - sqrt({d}))/{2*a}")
            print(f"r1 = {eqn1}")
            print(f"r2 = {eqn2}")
            final = sp.simplify((f"C1*exp(t*({eqn1})) + C2*exp(t*({eqn2}))"))
            print(f"The complimentary solution is \ny(t) = {expand(final)}")
            
        else:
            print("r = (-b \u00B1 sqrt(d))/(2*a) ")
            print((f"r = ({-b} \u00B1 sqrt({d}))/(2*{a})"))
            print(f"r = ({-b} \u00B1 {sp.sympify(f'sqrt({d})')})/{sp.sympify(2*a)}")
            print(f"r = {-b/(2*a)} \u00B1 {sp.sympify(f'sqrt({d})/{(2*a)}')}")
            alpha, beta = symbols('alpha beta')
            alpha,beta = pretty(alpha, use_unicode=True), pretty(beta, use_unicode=True)
            alp= -b/(2*a)
            bet = sp.sympify(f'sqrt({-d})/{(2*a)}')
            print("Therefore, ")
            print(f"{alpha} = {alp}, {beta} = {bet}")
            print(f"y(t) = exp(({alpha}*t)*(C1*sin({beta}*t) + C2*cos({beta}*t))")
            if alp == 0:
                final = sp.sympify(f"C1*sin({bet}*t) + C2*cos({bet}*t)")
            else:
                final = sp.sympify(f"(C1*sin({bet}*t) + C2*cos({bet}*t))")
                final = sp.simplify(sp.sympify(f"exp(({alp})*t)") * final)
            print(f"The complimentary solution is \ny(t) = {expand(final)}")

    else:
        print(f"r = -b/(2*a)")
        print(f"r = {-b}/({2*a})")
        r = sp.simplify(-b/(2*a))
        print(f"r = {r}")
        print("y(t) = exp(r*t)*(C1 + C2*t)")
        final = sp.sympify(f'exp({r}*t)*(C1 + C2*t)')
        print(f"The complimentary solution is \ny(t) = {expand(final)}")
    if p_x==0 or p_x=="0":
        print("Since it is homogeneous, there is no particular solution \n y_p(t) = 0")
        print("y(t) = y_comp + y_part")
        print(f"y(t) = {expand(final)}")
    else:
        comp = f"{expand(final)}"
        p_x = f"{p_x}"
        print(f"Since it is non-homogenous, there is a particular solution for {p_x}")
        y_part= get_part_special(a,b,c,p_x,comp) 
        print("y(t) = y_comp + y_part")
        final = final + y_part
        print(f"y(t) = {expand(sp.simplify(final))}")
    return expand(final)

#Second order IVP solver with steps
def second_order_solver_ivp(a,b,c,p_x, t1, t2, q1, q2):
    C1 = sp.symbols('C1')
    C2 = sp.symbols('C2')
    p_x_copy = p_x
    print(f"({a})*y`` + ({b})*y` + ({c})*y = {p_x}")
    print("a*y`` + b*y` +c*y = (p(t))")
    print("Comparing Coefficients...")
    print(f"a = {a}, b = {b}, c = {c}, p(t) = {p_x}")
    print(f"Therefore the complementary equation is: \n{a}*r**(2) + {b}*r + {c} = 0")
    print("d = b**(2) - (4*a*c)")
    print(f"d = {b}**2 - (4*{a}*{c})")
    print(f"d = {b**2} - {4*a*c}")
    d = b**(2) - (4*a*c)
    print(f"d = {d}")
    if d!=0:
        if d>0:
            print("r = (-b \u00B1 sqrt(d))/(2*a) ")
            print((f"r = ({-b} \u00B1 sqrt({d}))/(2*{a})"))
            print(f"r = ({-b} \u00B1 {sp.sympify(f'sqrt({d})')})/{sp.sympify(2*a)}")
            eqn1 = sp.sympify(f"({-b} + sqrt({d}))/{2*a}")
            eqn2 = sp.sympify(f"({-b} - sqrt({d}))/{2*a}")
            print(f"r1 = {eqn1}")
            print(f"r2 = {eqn2}")
            final = sp.simplify(f"C1*exp(t*({eqn1})) + C2*exp(t*({eqn2}))")
            print(f"Therefore, \ny(t) = {final}")
        else:
            print("r = (-b \u00B1 sqrt(d))/(2*a) ")
            print((f"r = ({-b} \u00B1 sqrt({d}))/(2*{a})"))
            print(f"r = ({-b} \u00B1 {sp.sympify(f'sqrt({d})')})/{sp.sympify(2*a)}")
            print(f"r = {-b/(2*a)} \u00B1 {sp.sympify(f'sqrt({d})/{(2*a)}')}")
            alpha, beta = symbols('alpha beta')
            alpha,beta = pretty(alpha, use_unicode=True), pretty(beta, use_unicode=True)
            alp= -b/(2*a)
            bet = sp.sympify(f'sqrt({-d})/{(2*a)}')
            print("Therefore, ")
            print(f"{alpha} = {alp}, {beta} = {bet}")
            print(f"y(t) = exp({alpha}*t)*(C1*sin({beta}*t) + C2*cos({beta}*t))")
            if alp == 0:
                final = f"C1*sin({bet}*t) + C2*cos({bet}*t)"
            else:
                final = sp.simplify(f"(C1*sin({bet}*t) + C2*cos({bet}*t))")
            print(f"y(t) = exp({alp}*t)*"+f"({final})")
            final = sp.simplify(f"exp({alp}*t)*"+f"({final})")
    else:
        print(f"r = -b/(2*a)")
        print(f"r = {-b}/({2*a})")
        r = sp.simplify(-b/(2*a))
        print(f"r = {r}")
        print("y(t) = exp(r*t)*(C1 + C2*t)")
        print(f"y(t) = {sp.simplify(f'exp({r}*t)*(C1 + C2*t)')}")
        final = sp.simplify(f'exp({r}*t)*(C1 + C2*t)')
    if p_x=="0":
        print("Since it is homogeneous, there is no particular solution \n y_p(t) = 0")
        print("y(t) = y_comp + y_part")
        print(f"y(t) = {expand(final)}")
    else:
        comp = f"{final}"
        print(f"Since it is non-homogenous, there is a particular solution for {p_x}")
        y_part= get_part_special(a,b,c,p_x_copy,comp) 
        print("y(t) = y_comp + y_part")
        final = final + y_part
        print(f"y(t) = {expand(sp.simplify(final))}")
    print(f"y`(t) = {final.diff(t)}")
    sub1 = final.subs(t,t1)
    ivp1 = Eq(sub1, q1)
    sub2 = final.diff(t,1).subs(t,t2)
    ivp2 = Eq(sub2, q2)
    print("Substituting the IVP Values")
    print(f"y({t1}) = {sub1} = {q1}")
    print(f"y`({t2}) = {sub2} = {q2}")
    solution = solve((ivp1, ivp2), (C1,C2))
    c1 = solution[C1]
    c2 = solution[C2]
    print(f"Upon solving the simultaneous equation, \nC1 = {c1}, C2 = {c2} ")
    print(f"The solution of the differential equation {a})*y`` + ({b})*y` + ({c})*y = {p_x}, y({t1}) = {q1}, y`({t2}) = {q2} is ")
    print(f"{expand(sp.simplify(final.subs(C1,c1).subs(C2,c2)))}")
    return (expand(sp.simplify(final.subs(C1,c1).subs(C2,c2))))

#Second Order BVP solver with steps
def second_order_solver_bvp(a,b,c,p_x, t1, t2, q1, q2):
    t1, t2 = sp.sympify(f"{t1}"), sp.sympify(f"{t2}")
    q1, q2 = sp.sympify(f"{q1}"), sp.sympify(f"{q2}")
    C1 = sp.symbols('C1')
    C2 = sp.symbols('C2')
    p_x_copy = p_x
    print(f"({a})*y`` + ({b})*y` + ({c})*y = {p_x}")
    print("a*y`` + b*y` +c*y = (p(t))")
    print("Comparing Coefficients...")
    print(f"a = {a}, b = {b}, c = {c}, p(t) = {p_x}")
    print(f"Therefore the complementary equation is: \n{a}*r**(2) + {b}*r + {c} = 0")
    print("d = b**(2) - (4*a*c)")
    print(f"d = {b}**2 - (4*{a}*{c})")
    print(f"d = {b**2} - {4*a*c}")
    d = b**(2) - (4*a*c)
    print(f"d = {d}")
    if d!=0:
        if d>0:
            print("r = (-b \u00B1 sqrt(d))/(2*a) ")
            print((f"r = ({-b} \u00B1 sqrt({d}))/(2*{a})"))
            print(f"r = ({-b} \u00B1 {sp.sympify(f'sqrt({d})')})/{sp.sympify(2*a)}")
            eqn1 = sp.sympify(f"({-b} + sqrt({d}))/{2*a}")
            eqn2 = sp.sympify(f"({-b} - sqrt({d}))/{2*a}")
            print(f"r1 = {eqn1}")
            print(f"r2 = {eqn2}")
            final = sp.simplify(f"C1*exp(t*({eqn1})) + C2*exp(t*({eqn2}))")
            print(f"Therefore, \ny(t) = {expand(final)}")
        else:
            print("r = (-b \u00B1 sqrt(d))/(2*a) ")
            print((f"r = ({-b} \u00B1 sqrt({d}))/(2*{a})"))
            print(f"r = ({-b} \u00B1 {sp.sympify(f'sqrt({d})')})/{sp.sympify(2*a)}")
            print(f"r = {-b/(2*a)} \u00B1 {sp.sympify(f'sqrt({d})/{(2*a)}')}")
            alpha, beta = symbols('alpha beta')
            alpha,beta = pretty(alpha, use_unicode=True), pretty(beta, use_unicode=True)
            alp= -b/(2*a)
            bet = sp.sympify(f'sqrt({-d})/{(2*a)}')
            print("Therefore, ")
            print(f"{alpha} = {alp}, {beta} = {bet}")
            print(f"y(t) = exp({alpha}*t)*(C1*sin({beta}*t) + C2*cos({beta}*t))")
            if alp == 0:
                final = sp.simplify(f"C1*sin({bet}*t) + C2*cos({bet}*t)")
            else:
                final = sp.simplify(f"exp({alp}*t)*(C1*sin({bet}*t) + C2*cos({bet}*t))")
                print(f"y(t) = exp({alp})*"+f"({final})")
            print(f"y(t) = {expand(final)}")
    else:
        print(f"r = -b/(2*a)")
        print(f"r = {-b}/({2*a})")
        r = sp.simplify(-b/(2*a))
        print(f"r = {r}")
        print("y(t) = exp(r*t)*(C1 + C2*t)")
        print(f"y(t) = {sp.simplify(f'exp({r}*t)*(C1 + C2*t)')}")
        final = sp.sympify(f'exp({r}*t)*(C1 + C2*t)')
    if p_x==0:
        print("Since it is homogeneous, there is no particular solution \n y_p(t) = 0")
        print("y(t) = y_comp + y_part")
        print(f"y(t) = {expand(final)}")
    else:
        comp = f"{final}"
        print(f"Since it is non-homogenous, there is a particular solution for {p_x}")
        y_part= get_part_special(a,b,c,p_x_copy,comp)  
        print("y(t) = y_comp + y_part")
        final = final + y_part
        print(f"y(t) = {expand(sp.simplify(final))}")
    try:
        sub1 = final.subs(t,t1)
        ivp1 = Eq(sub1, q1)
        sub2 = final.subs(t,t2)
        ivp2 = Eq(sub2, q2)
        print("Substituting the BVP Values")
        print(f"y({t1}) = {sub1} = {q1}")
        print(f"y({t2}) = {sub2} = {q2}")
        solution = solve((ivp1, ivp2), (C1,C2))
        c1, c2 = solution[C1],solution[C2]
        print(f"C1 = {c1}, C2 = {c2}")
        print(f"The solution of the differential equation {a})*y`` + ({b})*y` + ({c})*y = {p_x}, y({t1}) = {q1}, y({t2}) = {q2} is ")
        print(f"{expand(sp.simplify(final.subs(C1,c1).subs(C2,c2)))}")
        return (sp.simplify(final.subs(C1,c1).subs(C2,c2)))
    except:
        print("There is no solution")

#This is the third order equation solver
def third_order():
    order=3
    print("a*y``` + b*y`` + c*y` + d*y = p(x)")
    a=sp.sympify(input("Enter the value of a: "))
    b=sp.sympify(input("Enter the value of b: "))
    c=sp.sympify(input("Enter the value of c: "))
    d=sp.sympify(input("Enter the value of d: "))
    p_x=input("Enter the value of p(x): ")
    if p_x!=0:
        p_x=sp.sympify(p_x)
    equation=Eq(a*y(t).diff(t,3) + b*y(t).diff(t,2) + c*y(t).diff(t) + d*y(t), p_x)
    try:
        solution = dsolve(equation, y(t)).rhs
        print(f"The solution to the differential equation {a}*y``` + {b}*y`` + {c}*y` + {d}*y = {p_x} is: ")
        return solution
    except:
        solution = "Can't solve"
        print(solution)

#This is the fourth order equation solver
def fourth_order():
    order=4
    print("a*y```` + b*y``` + c*y`` + d*y` +e*y = p(x)")
    a=sp.sympify(input("Enter the value of a: "))
    b=sp.sympify(input("Enter the value of b: "))
    c=sp.sympify(input("Enter the value of c: "))
    d=sp.sympify(input("Enter the value of d: "))
    e=sp.sympify(input("Enter the value of e: "))
    p_x=input("Enter the value of p(x): ")
    p_x=sp.sympify(p_x)
    equation=Eq(a*y(t).diff(t,4) + b*y(t).diff(t,3) + c*y(t).diff(t,2) + d*y(t).diff(t) + e*y(t), p_x)
    try:
        solution = dsolve(equation, y(t)).rhs
        print(f"The solution to the differential equation {a}*y```` + {b}*y``` + {c}*y`` + {d}*y` + {e}*y = {p_x} is: ")
        return expand(sp.simplify(solution))
    except:
        print("Can't solve")

def fourth_order_ivp():
    # Define the order of the differential equation
    order = 4
    
    # Print the form of the differential equation
    print("a*y'''' + b*y''' + c*y'' + d*y' + e*y = p(x)")
    
    # Input coefficients and function
    a = sp.sympify(input("Enter the value of a: "))
    b = sp.sympify(input("Enter the value of b: "))
    c = sp.sympify(input("Enter the value of c: "))
    d = sp.sympify(input("Enter the value of d: "))
    e = sp.sympify(input("Enter the value of e: "))
    p_x = sp.sympify(input("Enter the value of p(x): "))
    
    # Define the independent variable and the function
    t = symbols('t')
    y = Function('y')
    
    # Define the differential equation
    equation = Eq(a * y(t).diff(t, 4) + b * y(t).diff(t, 3) + c * y(t).diff(t, 2) + d * y(t).diff(t) + e * y(t), p_x)
    
    # Input initial conditions
    y0 = sp.sympify(input("Enter the initial condition y(0): "))
    y1 = sp.sympify(input("Enter the initial condition y'(0): "))
    y2 = sp.sympify(input("Enter the initial condition y''(0): "))
    y3 = sp.sympify(input("Enter the initial condition y'''(0): "))
    
    # Define the initial conditions
    ics = {y(0): y0, y(t).diff(t).subs(t, 0): y1, y(t).diff(t, 2).subs(t, 0): y2, y(t).diff(t, 3).subs(t, 0): y3}
    try:
        # Solve the differential equation with initial conditions
        solution = dsolve(equation, y(t), ics=ics).rhs
        
        # Print the solution
        print(f"The solution to the differential equation {a}*y'''' + {b}*y''' + {c}*y'' + {d}*y' + {e}*y = {p_x} with initial conditions y(0)={y0}, y'(0)={y1}, y''(0)={y2}, y'''(0)={y3} is: ")
        return expand(sp.simplify(solution))
    except Exception as e:
        print("Can't solve the differential equation.")
    
#This is the fifth order equation solver
def fifth_order():
    order=5
    print("a*y````` + b*y```` + c*y``` + d*y`` +e*y` + f*y = p(x)")
    a=sp.sympify(input("Enter the value of a: "))
    b=sp.sympify(input("Enter the value of b: "))
    c=sp.sympify(input("Enter the value of c: "))
    d=sp.sympify(input("Enter the value of d: "))
    e=sp.sympify(input("Enter the value of e: "))
    f=sp.sympify(input("Enter the value of f: "))
    p_x=input("Enter the value of p(x): ")
    p_x=sp.sympify(p_x)
    equation=Eq(a*y(t).diff(t,5) + b*y(t).diff(t,4) + c*y(t).diff(t,3) + d*y(t).diff(t,2) + e*y(t).diff(t) + f*y(t), p_x)
    try:
        solution = dsolve(equation, y(t)).rhs
        print(f"The solution to the differential equation {a}*y````` + {b}*y```` + {c}*y``` + {d}*y`` + {e}*y` + {f}*y= {p_x} is: ")
        return expand(sp.simplify(solution))
    except:
        print("Can't solve")

def sixth_order():
    order=6
    print("a*y`````` + b*y````` + c*y```` + d*y``` +e*y`` + f*y` + g*y = p(x)")
    a=sp.sympify(input("Enter the value of a: "))
    b=sp.sympify(input("Enter the value of b: "))
    c=sp.sympify(input("Enter the value of c: "))
    d=sp.sympify(input("Enter the value of d: "))
    e=sp.sympify(input("Enter the value of e: "))
    f=sp.sympify(input("Enter the value of f: "))
    g=sp.sympify(input("Enter the value of g: "))
    p_x=input("Enter the value of p(x): ")
    p_x=sp.sympify(p_x)
    equation=Eq(a*y(t).diff(t,6) + b*y(t).diff(t,5) + c*y(t).diff(t,4) + d*y(t).diff(t,3) + e*y(t).diff(t,2) + f*y(t).diff(t) + g*y(t), p_x)
    try:
        solution = dsolve(equation, y(t)).rhs
        print(f"The solution to the differential equation {a}*y`````` + {b}*y````` + {c}*y```` + {d}*y``` + {e}*y`` + {f}*y` + {g}*y= {p_x} is: ")
        return expand(sp.simplify(solution))
    except:
        print("Can't solve")

def main():
    order= int(input("Enter the order of the differential equation: "))
    if order>0 and order<=6:
        if order==1:
            a,b, p_x, ics, solve, q, solution = first_order()
            choice = input("Would you like the steps to the solution: ")
            if choice == "yes":
                print("\n \n")
                if ics == "y":
                    return first_order_solver_ivp(a,b,p_x,solve,q)
                elif ics == "n":
                    return first_order_solver(a,b,p_x)
            else:
                return solution
        elif order==2:
            a,b,c,p_x, steps, ics, typee, solve1, solve2, q1, q2, solution = second_order()
            if steps:
                choice = input("would you like the steps to the solutions: ")
                if choice == "yes":
                    print("\n \n")
                    if ics =="y":
                        if typee.upper() == "IVP":
                            return second_order_solver_ivp(a,b,c,p_x,solve1, solve2, q1,q2)
                        elif typee.upper() == "BVP":
                            return second_order_solver_bvp(a,b,c,p_x,solve1, solve2, q1,q2)
                    else:
                        return second_order_solver(a,b,c,p_x)
                else:
                    return solution
            else:
                return solution
            
                
        elif order==3:
            return third_order()
        elif order==4:
            return fourth_order()
        elif order==5:
            return fifth_order()
        elif order==6:
            return sixth_order()
    else:
        print("Invalid Input")
main()
