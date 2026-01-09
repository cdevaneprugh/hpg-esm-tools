# Claude Code To Do List
Document Purpose: Serve as a to-do list for the user and Claude Code to reference when reorganizing the user's workspace,
refactoring scripts, revising documentation, creating slash commands, and creating agents.

## Organizing Workspace
The user is currently reorganizing his workspace as his scripts, docs, and files had become scattered over time.
1. Set up new git repos.
    - One for the user's /blue space (excluding earth model source code, hpg-esm documentation, and dot folders with the exception of Claude)
    - The cloned model will be addressed later.
    - One for the hpg-esm documentation that will be setup with github pages, MkDocs, and the Material theme.
2. Look over existing scripts and docs.
    - This is where Claude Code will read everything and assess the next steps.
    - In the scattered docs I'd like to remove redundant information, and update/keep relevant information. Leave the hpg-esm.documentation ALONE for now.
    - Work with the user to determine which scripts will be saved or removed, which will be merged, which need to be refactored (or made pretty), and which could be turned into Claude Code Agents or Slach Commands.

## Conda Environment setup
Default environment should include linters, formatters, and other syntax error checkers that Claude Code could 
leverage when designing and implenenting code.

Claude Code will activelly participate in this process as it is will be the primary user of these tools.

Packages:
- NCO tools
- Ferret
- Python tools for loading and reading netcdf files

## Refactoring
1. Based on the previous planning steps, Claude Code can refactor and make existing scripts pretty.
    - Create a consistent format for things like plotting scripts.
    - Add comments to existing code.
    - Create and revise documentation for code
2. Some scripts could be replaced by Agents & Slash commands.
    - I would like answered: is any shunk of the script or function useful to keep for Claude to leverage when calling agents?
    - Would just a clean agent document based on a script analysis suffice?

## Agents & Slash Commands
After existing scripts and docs have been cleaned up and refactored, and documentation has been revised,  we will move onto creating new tools for Claude to leverage. Some of these may be redundant depending on what we did during organization and refactor.

### Agents
1. Case Troubleshooter/Debugger
    * Takes case directory as input, reads logs, figures out why case failed.
    * Fixes it, resubmits case and monitors output **I'm unsure about this function**

2. Hist file and case parser
    * Could replace case_analyzer scripts (or at the very least leverage them)
    * Query case variables
    * Examine and report on hist files generated

3. CTSM Module analyzer
    * Dig up a specific parameter or submodule that's buried in the CTSM source code
    * I see limited use for this as of now, normally I'd want to keep module information in the working context window

4. Documentation reconciler
    * Reconcile discrepencies between online docs, local READMEs, and code observation
    * Could also generate a file that details solutions, observations, and insights
    * This is high priority as it will be needed for generating extensive CTSM docs

5. Case creator
    * Takes an input compset, grid, specific variables, and creates and builds a case.
    * Maybe add function to monitor case until it successfully starts executing?

6. Paper summarizer and info extractor.

### Slash Commands
1. Case Troubleshooter/Debugger
    * Takes case directory as input, reads logs, figures out why case failed.
    * Functionally the same as the agent, but for when I want all the context kept in the chat.

2. Current project status on github
    * Look at newest project commits and compare to our version
    * Any notable upgrades for our use case?
    * Determine if it is worth updating to a newer tag or incorporating changes made to specific files
    * Give an analysis and recommendation, detailing advantages and possible issues

3. Case creator
    * A command that calls an agent, creates and builds a ctsm case.
    * This may be unnecessary as an agent may suffice.

## Claude Specific Tasks
After we "retool" and upgrade the system, there are specific tasks to tackle with Claude Code.

1. Clone ctsm and setup w/ claude. See what decisions it makes. Any other ways to optimize? How does it setup the configs?
    * Would it be easier to use a conda environment to load all the packages needed by ctsm?

2. Analyze the best way to fork CTSM.
    * Fork main repo + cime config? Workaround?
    * Add claude files for tracking
    * Add user modified sourcecode directory
    * Maybe add a script directory for relevant user scripts?
    * Help managing all the gitmodules and configs

3. Which version is the best for us to use based on our needs?
    * Existing works fine, the user wants to know what has been changed in the repo and if it is worth upgrading.
    * A good opportunity to test out the new Slash commands or an agent that analyzes changes in the main repo, open issues, etc.

4. Update bashrc and vimrc. Look for better ways to do things.
    - unnecessary commands
    - is anything I have currently defined "wrong" or a bad practice?
    - vertical marker at space 80? - For code best practices (this might be legacy guidance and no longer relevant)
    - vim plugins or switch to neovim (I need a syntax highlighter, spellcheck, autocomplete)

5. Claude Code best practices and configs.
    - NCAR and CESM provide resources in the forms of wikis and pdf on the best practices for cesm/ctsm development.
    - You will help me pull that documentation from scattered sources online, parse what is relevant and out of date, and implement a guid for Claude Code to use when developing source code.
    - Local summaries and providing links to sources is probably the best way to go here.
    - Setting up an order of importance for references might also be a good idea (1. local summaries, 2. sources provided, 3. forums)
    - The CESM forums are a great resource, we need to make sure to mention this in the list of places to consult on anything cesm related.

6. Deep dive into the CTSM model and source code.
    * Documentation is conflicting and horribly out of date in places.
    * Work with the user to comb through and figure out: what docs are broken, what's up to date, what's useful even though it's technically out of date, etc
    * Write your own documentation to reference
    * We don't need to go through every single file, but we should carefully look at commonly used scripts, source code, and build directions

7. Use all information gained in previous steps to help build out and revise the hpg-esm.documentation.
    * This will serve as a guide for future researchers and students.
    * Goal will be to work with the user on the design and implementation of all the information gathered and integrate with existing docs.
    * Create online documentation with MkDocs and github pages

8. X11 setup. The X11 forwarding for viewing plots is a bit crude. Help me figure out a way to streamline this. - low priority

9. Reconcile inputdata files used by the run tower script.
    * Goal is to use every local piece of data possible.
    * low priority - do last

10. The input data for the earth models is "owned" by me because I downloaded it.
    - This will be a problem when I leave this job as my personal data will be deleted.
    - Can we change the ownership recursivelly to the group leader?
    - I'd like to check and make sure the permissions are set up correctly too. IIRC we use a lustre file system and the group has run into permission issues in the past where the chmod command doesn't change the lustre file permissions.
