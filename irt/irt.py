from builtins import range
from numpy import abs, array, concatenate, copy, exp, log, max, sort
from numpy.random import normal, uniform
from scipy.optimize import minimize
from scipy.stats import norm, halfnorm
from scipy.special import expit

__all__ = ['two_parameter_model', 'four_parameter_model', 'estimate_thetas']

def scale_guessing(value, c, d):
    return c + (d - c) * value

def two_parameter_model(a, b, theta):
    return expit(a * theta + b)

def four_parameter_model(a, b, c, d, theta):
    return scale_guessing(two_parameter_model(a, b, theta), c, d)

def learn_theta(abcds, corrects):
    def f(theta):
        theta = theta[0]
        mult = -norm(0, 1).pdf(theta)
        for abcd, correct in zip(abcds, corrects):
            a,b,c,d = abcd
            if correct != 0:
                p = four_parameter_model(a, b, c, d, theta)
                mult *= 1 + (2 * p - 1) * correct
        return mult
    return f

def learn_abcd(thetas, corrects):
    def f(arg):
        a, b, c, d = arg
        if a <= 0:
            return 0
        mult = -norm(0, 1).pdf(log(a))
        mult *= norm(0, 1).pdf(b)
        mult *= halfnorm(0, 0.1).pdf(c)
        mult *= halfnorm(0, 0.1).pdf(1 - d)
        for theta, correct in zip(thetas, corrects):
            p = four_parameter_model(a, b, c, d, theta)
            mult *= 1 + (2 * p - 1) * correct
        return mult
    return f

def expanded_scores(score_matrix):
    answers = [list(set(score)) for score in score_matrix]
    subscores = []

    for i, question_scores in enumerate(score_matrix):
        best = len(answers[i])
        student_count = len(question_scores)
        subscores_per_student = []

        for student_score in question_scores:
            expanded = [1] * answers[i].index(student_score)
            if len(expanded) < best - 1:
                expanded = expanded + [-1]
                expanded = expanded + [0] * (best - 1 - len(expanded))
            subscores_per_student.append(expanded)

        subscores.append(array(subscores_per_student).T)
    return concatenate(subscores)

def parse_optimization_result(res):
    if not res['success'] and res['message'] != 'Maximum number of iterations has been exceeded.':
        raise RuntimeError("Optimization failed:\n" + repr(res))
    return res['x']

def initialize_random_values(students_count, subquestions_count):
    theta_values = normal(scale = 0.01, size = students_count)
    a_values = exp(normal(scale = 0.01, size = subquestions_count))
    b_values = normal(0.01, size = subquestions_count)
    c_values = uniform(0., 0.01, size = subquestions_count)
    d_values = uniform(0.99, 1., size = subquestions_count)
    return theta_values, array([a_values, b_values, c_values, d_values]).T

def student_theta_given_abcd(abcds, scores, inital_theta):
    to_minimize = learn_theta(abcds, scores)
    res = minimize(to_minimize, [inital_theta], method = 'Nelder-Mead')
    return parse_optimization_result(res)

def all_thetas_given_abcd(abcds, scores, thetas):
    return array([student_theta_given_abcd(abcds, scores[:, i], thetas[i])
                  for i in range(len(thetas))])

def question_abcd_given_theta(thetas, scores, initial_abcd):
    to_minimize = learn_abcd(thetas, scores)
    res = minimize(to_minimize, initial_abcd, method = 'Nelder-Mead')
    return parse_optimization_result(res)

def all_abcds_given_theta(thetas, scores, abcds):
    return array([question_abcd_given_theta(thetas, scores[i], abcds[i])
                  for i in range(len(abcds))])

def estimate_thetas(scores):
    # Even though we usually input the table as scores per student,
    # the analysis is easier for a table of scores per question:
    scores = scores.T
    questions_count, students_count = scores.shape
    answers_per_question = [sort(array(list(set(score)))) for score in scores]
    subquestions_per_question = [len(answers) for answers in answers_per_question]
    subquestions_count = sum(subquestions_per_question) - questions_count

    thetas, abcds = initialize_random_values(students_count, subquestions_count)
    expanded = expanded_scores(scores)

    old_abcds, old_thetas = copy(abcds), copy(thetas)
    diff = 1
    small_diffs_streak = 0
    iter_count = 0
    
    while iter_count < 100 and small_diffs_streak < 3:
        abcds = all_abcds_given_theta(thetas, expanded, abcds)
        thetas = all_thetas_given_abcd(abcds, expanded, thetas)
        
        diff = max([max(abs(old_abcds - abcds)), max(abs(old_thetas - thetas))])
        if diff < 0.001:
            small_diffs_streak += 1
        else:
            small_diffs_streak = 0
        
        old_abcds, old_thetas = copy(abcds), copy(thetas)
        iter_count += 1
        
    return thetas, abcds
