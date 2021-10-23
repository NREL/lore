# Bash script to manage launching the learning case study.
# See /mediation/management/commands/learning_case_study.py for more details.
#
# Instructions to run on Nova:
#   cd ~/lore
#   git pull
#   cd lorecash
#   chmod +x ./mediation/management/commands/learning_case_study.sh
#   conda activate loredash
#   ./mediation/management/commands/learning_case_study.sh
# Once finished
#   cd ~/lore/loredash
#   mkdir solution
#   mv *.jsonl solution
#   python manage.py learning_case_study --plot=solution
# To transfer `solution` to another machine via scp:
#   tar -c solution -f solution.tar
#   scp -i ~/.ssh/nova.ssh odow@nova.iems.northwestern.edu:/home/odow/lore/loredash/solution.tar solution.tar

run_case_study() {
    for i in 1 2 3 4 5 6 7 8 9 10
    do
        python manage.py learning_case_study $1 $2 $i
    done
}
export -f run_case_study

nohup python manage.py learning_case_study 0.00 0.00 1 &
nohup bash -c 'run_case_study 0.00 0.02' &
nohup bash -c 'run_case_study 0.00 0.04' &

nohup bash -c 'run_case_study 0.02 0.00' &
nohup bash -c 'run_case_study 0.02 0.02' &
nohup bash -c 'run_case_study 0.02 0.04' &

nohup bash -c 'run_case_study 0.04 0.00' &
nohup bash -c 'run_case_study 0.04 0.02' &
nohup bash -c 'run_case_study 0.04 0.04' &
